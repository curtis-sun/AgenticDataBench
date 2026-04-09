"""
Auto Profiler - Automatically detect file types and extract structural/content profiles.

Supports:
- Structured: CSV, TSV, ARFF
- Semi-structured: Excel (xlsx/xls)
- Markup: JSON, YAML, HTML, Markdown, TeX
- Text: TXT, LOG
- Database: SQLite
- Binary: NPY, NPZ, NetCDF, GeoPackage
- Compressed: GZ, ZIP (recursive)

Usage:
    profiler = AutoProfiler()
    profiles = profiler.profile_directory("/path/to/domain")
"""

import os
import re
import csv
import json
import sqlite3
import gzip
import zipfile
import tarfile
import shutil
import tempfile
from collections import defaultdict
from typing import List, Dict, Optional, Any, Tuple

from data_classes import FileProfile


# =============================================================================
# Extension → File Type Mapping
# =============================================================================

EXTENSION_MAP = {
    # Structured tabular
    ".csv": "csv",
    ".tsv": "csv",          # treated same as csv with tab delimiter
    ".arff": "arff",        # Weka Attribute-Relation File Format
    # Semi-structured tabular
    ".xlsx": "excel",
    ".xls": "excel",
    # Markup / structured text
    ".json": "json",
    ".jsonl": "json",       # JSON Lines
    ".ndjson": "json",      # Newline-delimited JSON
    ".yaml": "yaml",
    ".yml": "yaml",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".markdown": "markdown",
    ".tex": "tex",
    ".latex": "tex",
    # Plain text
    ".txt": "text",
    ".log": "log",
    ".text": "text",
    # Database
    ".db": "sqlite",
    ".sqlite": "sqlite",
    ".sqlite3": "sqlite",
    # Binary - numpy
    ".npy": "numpy",
    ".npz": "numpy",
    # Binary - scientific
    ".nc": "netcdf",
    ".cdf": "netcdf",
    ".nc4": "netcdf",
    # Binary - geo
    ".gpkg": "geo",
    ".geojson": "json",     # treat as JSON (GeoJSON)
    ".shp": "geo",
    ".shx": "geo_companion",    # shapefile spatial index
    ".dbf": "dbf",              # dBASE / shapefile attribute table
    ".prj": "geo_companion",    # shapefile projection definition
    ".cpg": "geo_companion",    # shapefile code page / encoding
    # Binary - columnar
    ".parquet": "parquet",
    ".pq": "parquet",
    # Compressed (will be decompressed and recursively profiled)
    ".gz": "compressed",
    ".zip": "compressed",
    ".tar": "compressed",
    ".tgz": "compressed",
}

# Extensions to scan for in a domain directory
ALL_SUPPORTED_EXTENSIONS = set(EXTENSION_MAP.keys())


# =============================================================================
# Optional Dependency Helpers
# =============================================================================

def _try_import(module_name: str):
    """Try to import a module, return None if not available."""
    try:
        import importlib
        return importlib.import_module(module_name)
    except ImportError:
        return None


# =============================================================================
# Auto Profiler
# =============================================================================

class AutoProfiler:
    """
    Automatically detect file types and extract profiles.
    
    Profiles contain:
    - Structure: schema, fields, dimensions, etc.
    - Read params: encoding, delimiter, header, etc.
    - Content summary: row count, sample, description
    """
    
    def __init__(self, max_sample_rows: int = 3, max_file_size_mb: int = 800,
                 max_sample_columns: int = 80, llm_client=None):
        """
        Args:
            max_sample_rows: Number of sample rows to include in profile
            max_file_size_mb: Skip files larger than this
            max_sample_columns: Max columns to include in sample markdown (avoids huge tables for wide CSVs)
            llm_client: Optional LLM client with .call_json(prompt) method.
                        If provided, enables LLM-enhanced profiling for:
                        - Semi-structured Excel (header/data region detection)
                        - Unstructured text (classification and structure understanding)
                        Falls back to rule-based profiling if None or if LLM call fails.
        """
        self.max_sample_rows = max_sample_rows
        self.max_file_size_mb = max_file_size_mb
        self.max_sample_columns = max_sample_columns
        self.llm = llm_client
    
    # =========================================================================
    # Main Entry Points
    # =========================================================================
    
    def profile_directory(self, dir_path: str) -> List[FileProfile]:
        """
        Profile all supported files in a directory (recursive).
        
        Returns list of FileProfile objects.
        """
        profiles = []
        
        for root, dirs, filenames in os.walk(dir_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for fname in sorted(filenames):
                if fname.startswith('.'):
                    continue
                
                ext = self._get_extension(fname)
                if ext not in EXTENSION_MAP:
                    continue
                
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, dir_path)
                
                # Skip files that are too large
                try:
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    if size_mb > self.max_file_size_mb:
                        profiles.append(FileProfile(
                            path=rel_path,
                            file_type=EXTENSION_MAP[ext],
                            error=f"File too large ({size_mb:.1f} MB > {self.max_file_size_mb} MB limit)"
                        ))
                        continue
                except OSError:
                    continue
                
                profile = self.profile_file(full_path, rel_path)
                profiles.append(profile)
        
        return profiles
    
    def profile_file(self, full_path: str, rel_path: str = None) -> FileProfile:
        """
        Profile a single file.
        
        Args:
            full_path: Absolute path to the file
            rel_path: Relative path for display (defaults to basename)
        """
        if rel_path is None:
            rel_path = os.path.basename(full_path)
        
        ext = self._get_extension(full_path)
        file_type = EXTENSION_MAP.get(ext, "unknown")
        
        # Dispatch to type-specific profiler
        profiler_map = {
            "csv": self._profile_csv,
            "arff": self._profile_arff,
            "excel": self._profile_excel,
            "json": self._profile_json,
            "yaml": self._profile_yaml,
            "html": self._profile_html,
            "markdown": self._profile_markdown,
            "tex": self._profile_tex,
            "text": self._profile_text,
            "log": self._profile_log,
            "sqlite": self._profile_sqlite,
            "numpy": self._profile_numpy,
            "netcdf": self._profile_netcdf,
            "geo": self._profile_geo,
            "parquet": self._profile_parquet,
            "dbf": self._profile_dbf,
            "geo_companion": self._profile_geo_companion,
            "compressed": self._profile_compressed,
        }
        
        profiler_fn = profiler_map.get(file_type)
        if profiler_fn is None:
            return FileProfile(path=rel_path, file_type="unknown",
                               error=f"No profiler for extension '{ext}'")
        
        try:
            return profiler_fn(full_path, rel_path)
        except Exception as e:
            return FileProfile(path=rel_path, file_type=file_type,
                               error=f"Profiling failed: {str(e)[:200]}")
    
    # =========================================================================
    # Sample table helper (avoid huge markdown for wide tables)
    # =========================================================================
    
    def _sample_df_for_markdown(self, df: "pd.DataFrame") -> "pd.DataFrame":
        """
        Limit rows and columns for to_markdown().
        Uses pattern-aware column selection: keeps all standalone (non-pattern) columns
        and picks a few representatives (first, middle, last) from each pattern group,
        so the sample shows structural variety without wasting space on repetitive columns.
        """
        out = df.head(self.max_sample_rows)
        if len(df.columns) > self.max_sample_columns:
            patterns = self._detect_column_patterns(list(df.columns))
            if patterns['patterns']:
                selected = set(patterns['standalone'])
                for p in patterns['patterns']:
                    members = [c for c in df.columns
                               if re.sub(r'\d+', '{N}', c) == p['pattern']]
                    if len(members) >= 3:
                        selected.update([members[0], members[len(members) // 2], members[-1]])
                    else:
                        selected.update(members)
                # Preserve original column order, cap at max_sample_columns
                selected_ordered = [c for c in df.columns if c in selected]
                out = out[selected_ordered[:self.max_sample_columns]]
            else:
                out = out.iloc[:, :self.max_sample_columns]
        return out
    
    # =========================================================================
    # CSV / TSV Profiler
    # =========================================================================
    
    def _profile_csv(self, full_path: str, rel_path: str) -> FileProfile:
        """
        Profile a CSV/TSV file.
        
        - Delimiter: auto-detected via csv.Sniffer
        - Header row: heuristic first, LLM-assisted when uncertain
        """
        import pandas as pd
        
        # Step 1: Read raw lines for analysis
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                raw_lines = []
                for i, line in enumerate(f):
                    raw_lines.append(line.rstrip('\n'))
                    if i >= 20:
                        break
        except Exception as e:
            return FileProfile(path=rel_path, file_type="csv",
                               error=f"Failed to read CSV: {str(e)[:200]}")
        
        if not raw_lines:
            return FileProfile(path=rel_path, file_type="csv",
                               error="Empty file")
        
        # Step 2: Detect delimiter
        delimiter = ","
        try:
            sample_text = "\n".join(raw_lines)
            dialect = csv.Sniffer().sniff(sample_text, delimiters=',\t|;')
            delimiter = dialect.delimiter
        except csv.Error:
            if full_path.endswith('.tsv'):
                delimiter = "\t"
        
        # Step 3: Detect header row (heuristic + optional LLM)
        header_row = self._detect_csv_header(raw_lines, delimiter, rel_path)
        
        # Step 4: Read with pandas
        read_params = {"delimiter": delimiter, "header": header_row}
        df = None
        for enc in ["utf-8", "latin-1"]:
            try:
                df = pd.read_csv(
                    full_path,
                    encoding=enc,
                    delimiter=delimiter,
                    header=header_row,
                    low_memory=False,
                    nrows=50000,
                )
                read_params["encoding"] = enc
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        
        if df is None:
            return FileProfile(path=rel_path, file_type="csv",
                               error="Failed to parse CSV")
        
        # Step 5: Build profile
        structure = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "num_rows": len(df),
            "num_columns": len(df.columns),
            "numeric_columns": list(df.select_dtypes(include=['number']).columns),
            "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
            "missing_values": df.isnull().sum().to_dict(),
        }
        
        self._compute_all_column_patterns(structure)
        
        # Full row count
        total_rows = len(df)
        try:
            with open(full_path, 'r', encoding=read_params.get("encoding", "utf-8"),
                      errors='replace') as f:
                total_rows = sum(1 for _ in f) - (1 if header_row is not None else 0)
        except Exception:
            pass
        
        # Sample: limit columns for high-dimensional data to avoid huge markdown (e.g. 20532 cols → ~1.5MB)
        sample_df = self._sample_df_for_markdown(df)
        if len(df.columns) > self.max_sample_columns:
            structure["sample_columns_truncated_to"] = self.max_sample_columns
        sample_md = sample_df.to_markdown(index=False)
        
        return FileProfile(
            path=rel_path,
            file_type="csv",
            structure=structure,
            read_params=read_params,
            row_count=total_rows,
            sample=sample_md,
            content_summary=f"CSV file with {total_rows} rows and {len(df.columns)} columns",
            dataframe=df,
        )
    
    def _detect_csv_header(
        self, raw_lines: list, delimiter: str, file_path: str
    ) -> int:
        """
        Detect CSV header row position.
        
        Strategy:
        1. Heuristic: compare first row vs second row (text vs numeric)
        2. If uncertain and LLM available: send first 15 lines to LLM
        3. Default: 0
        """
        if len(raw_lines) < 2:
            return 0
        
        # Heuristic check
        first_fields = raw_lines[0].split(delimiter)
        second_fields = raw_lines[1].split(delimiter)
        
        first_all_text = all(not self._looks_numeric(f) for f in first_fields)
        second_has_numeric = any(self._looks_numeric(f) for f in second_fields)
        
        if first_all_text and second_has_numeric:
            # Confident: first row is header
            return 0
        
        # Uncertain — try LLM
        if self.llm is not None:
            rows_text = ""
            for i, line in enumerate(raw_lines[:15]):
                rows_text += f"  Row {i}: {line}\n"
            
            prompt = f"""Analyze this CSV file to find the header row.

File: {file_path}
Delimiter: {repr(delimiter)}

First rows (0-indexed):
{rows_text}

Which row (0-based index) contains the column headers?
- Row 0 if the first row contains column names
- A higher number if there are title/metadata rows before the actual header
- null if there is no header row (all rows are data)

Respond in JSON format only:
{{
    "header_row": <int or null>,
    "reason": "<brief explanation>"
}}"""
            
            try:
                result = self.llm.call_json(prompt)
                if result and result.get("header_row") is not None:
                    hr = result["header_row"]
                    if isinstance(hr, int) and 0 <= hr < min(15, len(raw_lines)):
                        return hr
            except Exception:
                pass
        
        # Default
        return 0
    
    @staticmethod
    def _looks_numeric(s: str) -> bool:
        """Check if a string looks like a number."""
        s = s.strip().strip('"').strip("'")
        if not s:
            return False
        try:
            float(s.replace(',', ''))
            return True
        except ValueError:
            return False
    
    # =========================================================================
    # ARFF Profiler
    # =========================================================================
    
    def _profile_arff(self, full_path: str, rel_path: str) -> FileProfile:
        """
        Profile ARFF (Attribute-Relation File Format) file used by Weka.
        
        Parses @RELATION, @ATTRIBUTE declarations, and @DATA section.
        Uses scipy.io.arff if available, otherwise falls back to manual parsing.
        """
        import pandas as pd
        
        # Phase 1: Parse header to extract relation name and attribute metadata
        relation_name = None
        attributes = []  # list of {"name": str, "type": str, "nominal_values": list|None}
        header_lines = 0
        data_line_start = None
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                for line_no, line in enumerate(f):
                    stripped = line.strip()
                    if not stripped or stripped.startswith('%'):
                        continue
                    upper = stripped.upper()
                    if upper.startswith('@RELATION'):
                        relation_name = stripped[len('@RELATION'):].strip().strip("'\"")
                    elif upper.startswith('@ATTRIBUTE'):
                        attr_def = stripped[len('@ATTRIBUTE'):].strip()
                        attr_name, attr_type, nominal_vals = self._parse_arff_attribute(attr_def)
                        attributes.append({
                            "name": attr_name,
                            "type": attr_type,
                            "nominal_values": nominal_vals,
                        })
                    elif upper.startswith('@DATA'):
                        data_line_start = line_no + 1
                        header_lines = line_no + 1
                        break
        except Exception as e:
            return FileProfile(path=rel_path, file_type="arff",
                               error=f"Failed to read ARFF header: {str(e)[:200]}")
        
        if not attributes:
            return FileProfile(path=rel_path, file_type="arff",
                               error="No @ATTRIBUTE declarations found")
        
        # Phase 2: Load data into DataFrame
        df = None
        parse_warnings = []
        scipy_arff = _try_import("scipy.io.arff")
        
        if scipy_arff is not None:
            try:
                data, meta = scipy_arff.loadarff(full_path)
                df = pd.DataFrame(data)
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].apply(
                            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x
                        )
            except Exception as e:
                parse_warnings.append(f"scipy.io.arff failed, fallback to manual parser: {str(e)[:160]}")
                df = None
        
        if df is None:
            df, manual_warnings = self._parse_arff_data_manual(full_path, attributes, data_line_start)
            parse_warnings.extend(manual_warnings)
        
        if df is None:
            return FileProfile(path=rel_path, file_type="arff",
                               error="Failed to parse ARFF data section")

        # Normalize textual columns, especially nominal/string values, to avoid
        # leading/trailing-space mismatches (e.g. " yes" vs "yes").
        for attr in attributes:
            col = attr["name"]
            if col not in df.columns:
                continue
            if attr["type"] in ("nominal", "string", "date"):
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        
        # Phase 3: Build profile
        col_names = list(df.columns)
        numeric_cols = list(df.select_dtypes(include=['number']).columns)
        categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
        
        attr_summary = []
        for attr in attributes:
            desc = f"{attr['name']}: {attr['type']}"
            if attr['nominal_values']:
                vals = attr['nominal_values']
                if len(vals) <= 8:
                    desc += f" {{{', '.join(vals)}}}"
                else:
                    desc += f" ({len(vals)} values)"
            attr_summary.append(desc)
        
        structure = {
            "relation": relation_name,
            "attributes": attributes,
            "attribute_summary": attr_summary,
            "columns": col_names,
            "num_columns": len(col_names),
            "num_rows": len(df),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "missing_values": df.isnull().sum().to_dict(),
        }
        if parse_warnings:
            structure["parse_warnings"] = parse_warnings
        
        self._compute_all_column_patterns(structure)
        
        total_rows = len(df)
        if data_line_start is not None:
            try:
                with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                    total_rows = sum(
                        1 for i, line in enumerate(f)
                        if i >= data_line_start and line.strip() and not line.strip().startswith('%')
                    )
            except Exception:
                pass
        
        sample_md = self._sample_df_for_markdown(df).to_markdown(index=False)
        
        return FileProfile(
            path=rel_path,
            file_type="arff",
            structure=structure,
            read_params={"format": "arff", "relation": relation_name},
            row_count=total_rows,
            sample=sample_md,
            content_summary=f"ARFF file '{relation_name or '?'}' with {total_rows} rows "
                           f"and {len(col_names)} attributes",
            dataframe=df,
        )
    
    @staticmethod
    def _parse_arff_attribute(attr_def: str) -> Tuple[str, str, Optional[List[str]]]:
        """
        Parse a single @ATTRIBUTE definition.
        Returns (name, type_string, nominal_values_or_None).
        """
        # Handle quoted attribute names
        if attr_def.startswith(("'", '"')):
            quote = attr_def[0]
            end = attr_def.index(quote, 1)
            name = attr_def[1:end]
            rest = attr_def[end + 1:].strip()
        else:
            parts = attr_def.split(None, 1)
            name = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
        
        rest_upper = rest.upper().strip()
        nominal_values = None
        
        if rest.strip().startswith('{'):
            attr_type = "nominal"
            inner = rest.strip()[1:].rstrip('}')
            nominal_values = [v.strip().strip("'\"") for v in inner.split(',')]
        elif rest_upper in ('NUMERIC', 'REAL', 'INTEGER'):
            attr_type = rest_upper.lower()
        elif rest_upper == 'STRING':
            attr_type = "string"
        elif rest_upper.startswith('DATE'):
            attr_type = "date"
        elif rest_upper.startswith('RELATIONAL'):
            attr_type = "relational"
        else:
            attr_type = rest.strip() or "unknown"
        
        return name, attr_type, nominal_values
    
    def _parse_arff_data_manual(
        self, full_path: str, attributes: list, data_line_start: Optional[int]
    ) -> Tuple[Optional["pd.DataFrame"], List[str]]:
        """Fallback: manually parse @DATA rows into a DataFrame."""
        import pandas as pd
        
        if data_line_start is None:
            return None, ["No @DATA section found for manual ARFF parsing"]
        
        col_names = [a["name"] for a in attributes]
        expected_cols = len(col_names)
        rows = []
        warnings = []
        mismatch_count = 0
        trailing_trim_count = 0
        padded_count = 0
        truncated_count = 0
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f):
                    if i < data_line_start:
                        continue
                    stripped = line.strip()
                    if not stripped or stripped.startswith('%'):
                        continue
                    values = self._split_arff_data_line(stripped, expected_cols)

                    # Common dirty-data case: a trailing comma creates one extra empty column.
                    if len(values) > expected_cols and values[-1] == "":
                        while len(values) > expected_cols and values[-1] == "":
                            values = values[:-1]
                            trailing_trim_count += 1

                    if len(values) != expected_cols:
                        mismatch_count += 1
                        line_no = i + 1
                        original_len = len(values)
                        if len(values) < expected_cols:
                            values = values + [""] * (expected_cols - len(values))
                            padded_count += 1
                            if len(warnings) < 20:
                                warnings.append(
                                    f"Line {line_no}: {original_len} cols < {expected_cols}; padded missing trailing columns"
                                )
                        else:
                            extras = values[expected_cols:]
                            values = values[:expected_cols]
                            truncated_count += 1
                            if len(warnings) < 20:
                                extra_note = "all empty" if all(v.strip() == "" for v in extras) else "contains non-empty values"
                                warnings.append(
                                    f"Line {line_no}: {original_len} cols > {expected_cols}; truncated extras ({extra_note})"
                                )

                    rows.append(values)
                    if len(rows) >= 50000:
                        break
        except Exception as e:
            return None, [f"Manual ARFF parse failed: {str(e)[:200]}"]
        
        if not rows:
            return None, ["No ARFF data rows parsed from @DATA section"]
        
        df = pd.DataFrame(rows, columns=col_names)

        # Normalize text before missing-value replacement.
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

        df.replace("", pd.NA, inplace=True)
        df.replace('?', pd.NA, inplace=True)
        
        for attr in attributes:
            col = attr["name"]
            if col not in df.columns:
                continue
            if attr["type"] in ("numeric", "real", "integer"):
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if trailing_trim_count > 0:
            warnings.append(
                f"Auto-fixed trailing empty columns caused by trailing commas: {trailing_trim_count} time(s)"
            )
        if mismatch_count > 0:
            warnings.append(
                f"Column mismatch rows auto-handled: {mismatch_count} "
                f"(padded={padded_count}, truncated={truncated_count})"
            )

        return df, warnings
    
    @staticmethod
    def _split_arff_data_line(line: str, expected_cols: int) -> List[str]:
        """Split an ARFF data line, respecting quoted values."""
        values = []
        current = []
        in_quote = None
        
        for ch in line:
            if in_quote:
                if ch == in_quote:
                    in_quote = None
                else:
                    current.append(ch)
            elif ch in ("'", '"'):
                in_quote = ch
            elif ch == ',':
                values.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        values.append(''.join(current).strip())
        
        return values
    
    # =========================================================================
    # Excel Profiler
    # =========================================================================
    
    def _profile_excel(self, full_path: str, rel_path: str) -> FileProfile:
        """
        Profile Excel file: extract sheet info, merged cells, header position.
        
        LLM-enhanced: For semi-structured sheets (merged cells, title rows,
        metadata rows before actual data), uses LLM to detect the real header
        row and data start position. Falls back to pandas default (row 0) if
        LLM is unavailable or the sheet looks like a standard table.
        """
        import pandas as pd
        openpyxl = _try_import("openpyxl")
        
        structure = {"sheets": []}
        read_params = {}
        combined_rows = 0
        sample_text = ""
        dataframe = None  # Store the first/main sheet as DataFrame
        
        # ---- Phase 1: openpyxl metadata extraction ----
        sheet_raw_rows = {}    # sheet_name -> list of first 15 rows (raw values)
        sheet_merged = {}      # sheet_name -> list of merged cell ranges
        
        if openpyxl and full_path.endswith(('.xlsx', '.xlsm')):
            try:
                wb = openpyxl.load_workbook(full_path, read_only=False, data_only=True)
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    sheet_info = {
                        "name": sheet_name,
                        "dimensions": ws.dimensions,
                        "max_row": ws.max_row,
                        "max_column": ws.max_column,
                    }
                    
                    # Collect merged cell ranges
                    merged_ranges = [str(m) for m in ws.merged_cells.ranges]
                    if merged_ranges:
                        sheet_info["merged_cells"] = merged_ranges
                        sheet_merged[sheet_name] = merged_ranges
                    
                    # Read first 15 rows as raw values for semi-structure detection
                    raw_rows = []
                    for row_idx, row in enumerate(ws.iter_rows(max_row=15, values_only=True)):
                        raw_rows.append([
                            str(cell) if cell is not None else ""
                            for cell in row
                        ])
                        if row_idx >= 14:
                            break
                    sheet_raw_rows[sheet_name] = raw_rows
                    
                    structure["sheets"].append(sheet_info)
                wb.close()
            except Exception:
                pass
        
        # ---- Phase 2: Per-sheet LLM-enhanced header detection + pandas read ----
        try:
            xls = pd.ExcelFile(full_path)
            sheet_names = xls.sheet_names
            
            for sheet_name in sheet_names:
                try:
                    # Determine header row: LLM for semi-structured, default for clean
                    header_row = 0
                    llm_excel_info = None
                    
                    raw_rows = sheet_raw_rows.get(sheet_name, [])
                    is_semi_structured = self._detect_semi_structured_excel(
                        raw_rows, sheet_merged.get(sheet_name, [])
                    )
                    
                    if is_semi_structured and self.llm is not None:
                        llm_excel_info = self._llm_detect_excel_header(
                            raw_rows, sheet_name, rel_path
                        )
                        if llm_excel_info and llm_excel_info.get("header_row") is not None:
                            header_row = llm_excel_info["header_row"]
                    
                    df = pd.read_excel(
                        full_path, sheet_name=sheet_name,
                        header=header_row, nrows=50000,
                    )
                    
                    sheet_profile = {
                        "name": sheet_name,
                        "num_rows": len(df),
                        "num_columns": len(df.columns),
                        "columns": list(df.columns),
                        "dtypes": df.dtypes.astype(str).to_dict(),
                        "numeric_columns": list(df.select_dtypes(include=['number']).columns),
                        "categorical_columns": list(df.select_dtypes(include=['object', 'category']).columns),
                        "missing_values": df.isnull().sum().to_dict(),
                    }
                    
                    self._compute_all_column_patterns(sheet_profile)
                    
                    if is_semi_structured:
                        sheet_profile["is_semi_structured"] = True
                        sheet_profile["detected_header_row"] = header_row
                        if llm_excel_info:
                            if llm_excel_info.get("metadata_summary"):
                                sheet_profile["metadata_summary"] = llm_excel_info["metadata_summary"]
                            if llm_excel_info.get("data_description"):
                                sheet_profile["data_description"] = llm_excel_info["data_description"]
                    
                    # Update or add to sheets list
                    found = False
                    for i, s in enumerate(structure["sheets"]):
                        if s["name"] == sheet_name:
                            structure["sheets"][i].update(sheet_profile)
                            found = True
                            break
                    if not found:
                        structure["sheets"].append(sheet_profile)
                    
                    combined_rows += len(df)
                    
                    # Use first sheet as main DataFrame and sample
                    if dataframe is None:
                        dataframe = df
                        sample_text = self._sample_df_for_markdown(df).to_markdown(index=False)
                        read_params = {"sheet_name": sheet_name, "header": header_row}
                
                except Exception:
                    continue
            
            xls.close()
        except Exception as e:
            if not structure["sheets"]:
                return FileProfile(path=rel_path, file_type="excel",
                                   error=f"Failed to read Excel: {str(e)[:200]}")
        
        # Also store columns/numeric at top level for backward compat
        if dataframe is not None:
            structure["columns"] = list(dataframe.columns)
            structure["num_rows"] = len(dataframe)
            structure["num_columns"] = len(dataframe.columns)
            structure["numeric_columns"] = list(
                dataframe.select_dtypes(include=['number']).columns)
            structure["categorical_columns"] = list(
                dataframe.select_dtypes(include=['object', 'category']).columns)
            structure["missing_values"] = dataframe.isnull().sum().to_dict()
        
        return FileProfile(
            path=rel_path,
            file_type="excel",
            structure=structure,
            read_params=read_params,
            row_count=combined_rows,
            sample=sample_text,
            content_summary=f"Excel file with {len(structure['sheets'])} sheet(s), "
                           f"{combined_rows} total rows",
            dataframe=dataframe,
        )
    
    def _detect_semi_structured_excel(
        self, raw_rows: list, merged_ranges: list
    ) -> bool:
        """
        Heuristic: decide whether a sheet is semi-structured and needs LLM
        help to find the real header row.
        
        Signals of semi-structured layout:
        1. Has merged cells in the first 5 rows (titles / section headers)
        2. First row has very few non-empty cells compared to later rows
        3. Empty rows exist in the first 10 rows (separators)
        4. First row content looks like a title (single long text string)
        """
        if not raw_rows or len(raw_rows) < 3:
            return False
        
        signals = 0
        
        # Signal 1: Merged cells in top area
        if merged_ranges:
            for mr in merged_ranges:
                # e.g. "A1:F1" — check if the row part is ≤ 5
                try:
                    row_num = int(re.search(r'(\d+)', mr).group(1))
                    if row_num <= 5:
                        signals += 2  # strong signal
                        break
                except Exception:
                    pass
        
        # Signal 2: First row has much fewer non-empty cells than later rows
        def non_empty_count(row):
            return sum(1 for c in row if c.strip())
        
        first_row_count = non_empty_count(raw_rows[0])
        later_counts = [non_empty_count(r) for r in raw_rows[2:min(8, len(raw_rows))]]
        if later_counts:
            avg_later = sum(later_counts) / len(later_counts)
            if first_row_count > 0 and avg_later > 0 and first_row_count < avg_later * 0.4:
                signals += 1
        
        # Signal 3: Empty rows in first 10 rows
        for row in raw_rows[:10]:
            if all(c.strip() == "" for c in row):
                signals += 1
                break
        
        # Signal 4: First row looks like a title (single non-empty cell, long text)
        if first_row_count == 1:
            for c in raw_rows[0]:
                if c.strip() and len(c.strip()) > 5:
                    signals += 1
                    break
        
        return signals >= 2
    
    def _llm_detect_excel_header(
        self, raw_rows: list, sheet_name: str, file_path: str
    ) -> Optional[Dict[str, Any]]:
        """
        Call LLM to detect header row and data region in a semi-structured
        Excel sheet.
        
        Sends the first 15 raw rows and asks LLM to identify:
        - header_row: 0-based index of the real column header row
        - data_start_row: 0-based index where data rows begin
        - metadata_summary: what the non-data rows contain
        - data_description: one-sentence description of the data
        
        Returns None if LLM is unavailable or call fails.
        """
        if not self.llm or not raw_rows:
            return None
        
        # Format raw rows into a readable table for the prompt
        rows_text = ""
        for i, row in enumerate(raw_rows):
            cells = " | ".join(row)
            rows_text += f"  Row {i}: [{cells}]\n"
        
        prompt = f"""Analyze this Excel sheet to find where the real data table starts.

File: {file_path}
Sheet: {sheet_name}

First rows (0-indexed):
{rows_text}

This sheet appears to have metadata/title rows before the actual data table.
Identify:
1. Which row (0-based index) contains the column headers?
2. Which row (0-based index) is where the data rows start (the row right after headers)?
3. What do the rows before the header contain? (brief summary)
4. What kind of data does this table contain? (one sentence)

Respond in JSON format only:
{{
    "header_row": <int, 0-based index of the header row>,
    "data_start_row": <int, 0-based index of first data row>,
    "metadata_summary": "<what the pre-header rows contain>",
    "data_description": "<one-sentence description of the data table>"
}}"""
        
        try:
            result = self.llm.call_json(prompt)
            if result and isinstance(result.get("header_row"), int):
                hr = result["header_row"]
                # Sanity check: header_row should be within the first 15 rows
                if 0 <= hr < min(15, len(raw_rows)):
                    return result
            return None
        except Exception:
            return None
    
    # =========================================================================
    # JSON Profiler
    # =========================================================================
    
    def _profile_json(self, full_path: str, rel_path: str) -> FileProfile:
        """
        Profile JSON/JSONL file: schema inference, structure analysis.
        """
        is_jsonl = full_path.endswith(('.jsonl', '.ndjson'))
        
        data = None
        records = []
        
        if is_jsonl:
            # Read first N lines for profiling
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue
                        records.append(json.loads(line))
                        if i >= 1000:
                            break
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                return FileProfile(path=rel_path, file_type="json",
                                   error=f"Failed to parse JSONL: {str(e)[:200]}")
            data = records
        else:
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                return FileProfile(path=rel_path, file_type="json",
                                   error=f"Failed to parse JSON: {str(e)[:200]}")
        
        # Analyze structure
        structure = self._analyze_json_structure(data)
        
        # Count records
        row_count = None
        if isinstance(data, list):
            row_count = len(data)
            if is_jsonl:
                # Count total lines for JSONL
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        row_count = sum(1 for line in f if line.strip())
                except Exception:
                    pass
        
        # Sample
        sample = ""
        if isinstance(data, list) and data:
            sample_data = data[:self.max_sample_rows]
            sample = json.dumps(sample_data, indent=2, ensure_ascii=False, default=str)[:2000]
        elif isinstance(data, dict):
            sample = json.dumps(
                {k: data[k] for k in list(data.keys())[:5]},
                indent=2, ensure_ascii=False, default=str
            )[:2000]
        
        # Try schema inference with genson
        schema = self._infer_json_schema(data)
        if schema:
            structure["schema"] = schema
        
        return FileProfile(
            path=rel_path,
            file_type="json",
            structure=structure,
            read_params={"format": "jsonl" if is_jsonl else "json"},
            row_count=row_count,
            sample=sample,
            content_summary=f"{'JSONL' if is_jsonl else 'JSON'} file, "
                           f"top-level type: {structure.get('top_level_type', '?')}"
                           + (f", {row_count} records" if row_count else ""),
        )
    
    def _analyze_json_structure(self, data: Any) -> Dict[str, Any]:
        """Analyze JSON structure: type, depth, keys."""
        structure = {}
        
        if isinstance(data, list):
            structure["top_level_type"] = "array"
            structure["record_count"] = len(data)
            if data and isinstance(data[0], dict):
                structure["top_level_keys"] = list(data[0].keys())
                # Check if all records have similar keys
                if len(data) > 1:
                    key_sets = [set(r.keys()) for r in data[:100] if isinstance(r, dict)]
                    if key_sets:
                        common = set.intersection(*key_sets)
                        all_keys = set.union(*key_sets)
                        structure["common_keys"] = list(common)
                        structure["all_keys"] = list(all_keys)
                        structure["schema_consistent"] = len(common) == len(all_keys)
        elif isinstance(data, dict):
            structure["top_level_type"] = "object"
            structure["top_level_keys"] = list(data.keys())[:50]
        else:
            structure["top_level_type"] = type(data).__name__
        
        structure["depth"] = self._json_depth(data, max_depth=10)
        
        return structure
    
    def _json_depth(self, obj: Any, current: int = 0, max_depth: int = 10) -> int:
        """Calculate nesting depth of JSON object."""
        if current >= max_depth:
            return current
        if isinstance(obj, dict):
            if not obj:
                return current + 1
            return max(self._json_depth(v, current + 1, max_depth) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current + 1
            # Sample first few elements only
            return max(self._json_depth(v, current + 1, max_depth) for v in obj[:5])
        return current
    
    def _infer_json_schema(self, data: Any) -> Optional[Dict]:
        """Infer JSON Schema using genson library."""
        genson = _try_import("genson")
        if genson is None:
            return None
        
        try:
            builder = genson.SchemaBuilder()
            if isinstance(data, list):
                for item in data[:50]:  # Sample first 50 records
                    builder.add_object(item)
            else:
                builder.add_object(data)
            return builder.to_schema()
        except Exception:
            return None
    
    # =========================================================================
    # YAML Profiler
    # =========================================================================
    
    def _profile_yaml(self, full_path: str, rel_path: str) -> FileProfile:
        """Profile YAML file: load and analyze as structured data."""
        try:
            import yaml
        except ImportError:
            return FileProfile(path=rel_path, file_type="yaml",
                               error="PyYAML not installed")
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            return FileProfile(path=rel_path, file_type="yaml",
                               error=f"Failed to parse YAML: {str(e)[:200]}")
        
        # Reuse JSON analysis (YAML loads into same Python types)
        structure = self._analyze_json_structure(data)
        
        row_count = len(data) if isinstance(data, list) else None
        
        sample = ""
        try:
            import yaml as _yaml
            if isinstance(data, list):
                sample = _yaml.dump(data[:self.max_sample_rows],
                                    default_flow_style=False, allow_unicode=True)[:2000]
            elif isinstance(data, dict):
                subset = {k: data[k] for k in list(data.keys())[:5]}
                sample = _yaml.dump(subset, default_flow_style=False,
                                    allow_unicode=True)[:2000]
        except Exception:
            sample = str(data)[:2000]
        
        return FileProfile(
            path=rel_path,
            file_type="yaml",
            structure=structure,
            read_params={"format": "yaml"},
            row_count=row_count,
            sample=sample,
            content_summary=f"YAML file, top-level type: {structure.get('top_level_type', '?')}",
        )
    
    # =========================================================================
    # HTML Profiler
    # =========================================================================
    
    def _profile_html(self, full_path: str, rel_path: str) -> FileProfile:
        """Profile HTML file: DOM structure, tables, headings."""
        bs4 = _try_import("bs4")
        import pandas as pd
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            return FileProfile(path=rel_path, file_type="html",
                               error=f"Failed to read HTML: {str(e)[:200]}")
        
        structure = {"elements": {}}
        sample = ""
        dataframe = None
        row_count = None
        
        if bs4:
            soup = bs4.BeautifulSoup(content, 'html.parser')
            
            # Extract headings
            headings = []
            for level in range(1, 7):
                for h in soup.find_all(f'h{level}'):
                    text = h.get_text(strip=True)[:100]
                    if text:
                        headings.append(f"h{level}: {text}")
            structure["headings"] = headings[:20]
            
            # Count key elements
            for tag in ['table', 'form', 'ul', 'ol', 'img', 'a', 'p']:
                count = len(soup.find_all(tag))
                if count > 0:
                    structure["elements"][tag] = count
            
            structure["tables_count"] = structure["elements"].get("table", 0)
        
        # Try to extract tables with pandas
        try:
            tables = pd.read_html(full_path)
            if tables:
                structure["tables_count"] = len(tables)
                structure["tables_info"] = []
                for i, t in enumerate(tables[:5]):
                    structure["tables_info"].append({
                        "index": i,
                        "rows": len(t),
                        "columns": list(t.columns),
                    })
                # Use first table as primary data
                dataframe = tables[0]
                row_count = len(dataframe)
                sample = self._sample_df_for_markdown(dataframe).to_markdown(index=False)
                
                # Add tabular structure for first table
                structure["columns"] = list(dataframe.columns)
                structure["numeric_columns"] = list(
                    dataframe.select_dtypes(include=['number']).columns)
        except Exception:
            pass
        
        # If no tables found, provide text sample
        if not sample:
            if bs4:
                text = soup.get_text(separator='\n', strip=True)
                sample = text[:1000]
                row_count = len(text.split('\n'))
            else:
                sample = content[:1000]
        
        return FileProfile(
            path=rel_path,
            file_type="html",
            structure=structure,
            read_params={},
            row_count=row_count,
            sample=sample,
            content_summary=f"HTML file with {structure.get('tables_count', 0)} table(s), "
                           f"{len(structure.get('headings', []))} heading(s)",
            dataframe=dataframe,
        )
    
    # =========================================================================
    # Markdown Profiler
    # =========================================================================
    
    def _profile_markdown(self, full_path: str, rel_path: str) -> FileProfile:
        """Profile Markdown file: heading structure, code blocks, tables."""
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            return FileProfile(path=rel_path, file_type="markdown",
                               error=f"Failed to read: {str(e)[:200]}")
        
        lines = content.split('\n')
        
        # Extract headings
        headings = []
        for line in lines:
            match = re.match(r'^(#{1,6})\s+(.+)', line)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headings.append(f"h{level}: {text}")
        
        # Count elements
        elements = {}
        code_blocks = len(re.findall(r'```', content)) // 2
        if code_blocks:
            elements["code_blocks"] = code_blocks
        
        # Detect tables (lines with |)
        table_lines = [l for l in lines if '|' in l and l.strip().startswith('|')]
        if table_lines:
            elements["table_rows"] = len(table_lines)
        
        # Detect lists
        list_items = len([l for l in lines if re.match(r'^\s*[-*+]\s', l) or re.match(r'^\s*\d+\.\s', l)])
        if list_items:
            elements["list_items"] = list_items
        
        # Detect links and images
        links = len(re.findall(r'\[.+?\]\(.+?\)', content))
        images = len(re.findall(r'!\[.*?\]\(.+?\)', content))
        if links:
            elements["links"] = links
        if images:
            elements["images"] = images
        
        structure = {
            "headings": headings[:20],
            "elements": elements,
            "line_count": len(lines),
            "char_count": len(content),
        }
        
        return FileProfile(
            path=rel_path,
            file_type="markdown",
            structure=structure,
            read_params={},
            row_count=len(lines),
            sample=content[:1500],
            content_summary=f"Markdown file with {len(lines)} lines, "
                           f"{len(headings)} heading(s)",
        )
    
    # =========================================================================
    # TeX Profiler
    # =========================================================================
    
    def _profile_tex(self, full_path: str, rel_path: str) -> FileProfile:
        """Profile LaTeX file: sections, environments, figures, tables."""
        try:
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            return FileProfile(path=rel_path, file_type="tex",
                               error=f"Failed to read: {str(e)[:200]}")
        
        lines = content.split('\n')
        
        # Extract section structure
        headings = []
        section_cmds = [
            (r'\\chapter\{(.+?)\}', "chapter"),
            (r'\\section\{(.+?)\}', "section"),
            (r'\\subsection\{(.+?)\}', "subsection"),
            (r'\\subsubsection\{(.+?)\}', "subsubsection"),
        ]
        for pattern, level in section_cmds:
            for match in re.finditer(pattern, content):
                headings.append(f"{level}: {match.group(1)}")
        
        # Count environments
        environments = {}
        env_pattern = r'\\begin\{(\w+)\}'
        for match in re.finditer(env_pattern, content):
            env_name = match.group(1)
            environments[env_name] = environments.get(env_name, 0) + 1
        
        structure = {
            "headings": headings[:20],
            "environments": environments,
            "elements": {
                "figures": environments.get("figure", 0),
                "tables": environments.get("table", 0) + environments.get("tabular", 0),
                "equations": environments.get("equation", 0) + environments.get("align", 0),
            },
            "line_count": len(lines),
        }
        
        return FileProfile(
            path=rel_path,
            file_type="tex",
            structure=structure,
            read_params={},
            row_count=len(lines),
            sample=content[:1500],
            content_summary=f"LaTeX file with {len(lines)} lines, "
                           f"{len(headings)} section(s)",
        )
    
    # =========================================================================
    # Text Profiler
    # =========================================================================
    
    def _profile_text(self, full_path: str, rel_path: str) -> FileProfile:
        """
        Profile plain text file: line count, structure detection.
        
        LLM-enhanced: When simple regex rules (key-value, fixed-width) fail to
        classify the text, calls LLM to identify the file type, describe its
        structure, and suggest how to process it. Falls back to basic stats if
        LLM is unavailable or the regex rules already produce a classification.
        """
        encoding = "utf-8"
        
        try:
            with open(full_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read(1_000_000)  # Read up to 1MB for profiling
        except Exception as e:
            return FileProfile(path=rel_path, file_type="text",
                               error=f"Failed to read: {str(e)[:200]}")
        
        lines = content.split('\n')
        
        structure = {
            "line_count": len(lines),
            "char_count": len(content),
            "encoding": encoding,
        }
        
        # ---- Rule-based detection (fast, no LLM needed) ----
        detected_by_rules = False
        if lines:
            sample_lines = lines[:20]
            
            # Check for key=value pattern
            kv_count = sum(1 for l in sample_lines if re.match(r'^[\w.]+\s*[=:]\s*', l))
            if kv_count > len(sample_lines) * 0.5:
                structure["detected_format"] = "key-value"
                detected_by_rules = True
            
            # Check for fixed-width columns
            if not detected_by_rules:
                widths = [len(l) for l in sample_lines if l.strip()]
                if widths and len(widths) >= 5 and max(widths) - min(widths) < 3:
                    structure["detected_format"] = "fixed-width"
                    detected_by_rules = True
        
        # ---- LLM-enhanced classification (when rules fail) ----
        if not detected_by_rules and self.llm is not None:
            llm_text_info = self._llm_classify_text(lines, rel_path)
            if llm_text_info:
                structure["detected_format"] = llm_text_info.get("file_type", "unknown")
                if llm_text_info.get("structure_description"):
                    structure["structure_description"] = llm_text_info["structure_description"]
                if llm_text_info.get("processing_suggestion"):
                    structure["processing_suggestion"] = llm_text_info["processing_suggestion"]
                if llm_text_info.get("has_tabular_data") is True:
                    structure["has_tabular_data"] = True
        
        content_summary = f"Text file with {len(lines)} lines"
        if "detected_format" in structure:
            content_summary += f" ({structure['detected_format']})"
        if "structure_description" in structure:
            content_summary += f" — {structure['structure_description']}"
        
        return FileProfile(
            path=rel_path,
            file_type="text",
            structure=structure,
            read_params={"encoding": encoding},
            row_count=len(lines),
            sample="\n".join(lines[:min(20, len(lines))]),
            content_summary=content_summary,
        )
    
    def _llm_classify_text(
        self, lines: list, file_path: str
    ) -> Optional[Dict[str, Any]]:
        """
        Call LLM to classify a text file that regex rules couldn't identify.
        
        Sends the first 40 lines and asks LLM to identify:
        - file_type: what kind of text this is
        - structure_description: how the content is organized
        - has_tabular_data: whether it contains table-like data
        - processing_suggestion: how a data agent should handle this file
        
        Returns None if LLM is unavailable or call fails.
        """
        if not self.llm or not lines:
            return None
        
        # Take first 40 non-empty lines for classification
        sample_lines = []
        for line in lines[:80]:
            if len(sample_lines) >= 40:
                break
            stripped = line.rstrip()
            if stripped:
                sample_lines.append(stripped)
        
        if not sample_lines:
            return None
        
        # Truncate each line to avoid huge prompts
        truncated = [l[:200] for l in sample_lines]
        sample_text = "\n".join(f"  {i:>3}: {l}" for i, l in enumerate(truncated))
        
        prompt = f"""Classify this text file and describe its structure.

File: {file_path}
Total lines: {len(lines)}

Sample (first ~40 non-empty lines):
{sample_text}

Identify:
1. What type of text file is this? (e.g., meeting_notes, config, data_dump, 
   code_output, email, report, chat_log, csv_like, xml_fragment, changelog, 
   readme, script_output, survey_response, or other)
2. How is the content structured? (one sentence)
3. Does it contain tabular or structured data that could be parsed into rows/columns?
4. How should a data analysis agent process this file?

Respond in JSON format only:
{{
    "file_type": "<type>",
    "structure_description": "<one-sentence structure description>",
    "has_tabular_data": true/false,
    "processing_suggestion": "<one-sentence suggestion for a data agent>"
}}"""
        
        try:
            result = self.llm.call_json(prompt)
            if result and isinstance(result.get("file_type"), str):
                return result
            return None
        except Exception:
            return None
    
    # =========================================================================
    # Log Profiler
    # =========================================================================
    
    def _profile_log(self, full_path: str, rel_path: str) -> FileProfile:
        """
        Profile log file: timestamp detection, log level distribution,
        template extraction.
        """
        encoding = "utf-8"
        
        try:
            with open(full_path, 'r', encoding=encoding, errors='replace') as f:
                lines = []
                for i, line in enumerate(f):
                    lines.append(line.rstrip('\n'))
                    if i >= 10000:
                        break
        except Exception as e:
            return FileProfile(path=rel_path, file_type="log",
                               error=f"Failed to read: {str(e)[:200]}")
        
        # Total line count
        total_lines = len(lines)
        try:
            with open(full_path, 'r', encoding=encoding, errors='replace') as f:
                total_lines = sum(1 for _ in f)
        except Exception:
            pass
        
        structure = {
            "line_count": total_lines,
            "is_log": True,
            "encoding": encoding,
        }
        
        # Detect timestamp patterns
        ts_patterns = [
            (r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', "ISO8601"),
            (r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}', "CLF"),
            (r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}', "syslog"),
            (r'\d{10,13}', "epoch"),
        ]
        
        sample_lines = lines[:100]
        for pattern, name in ts_patterns:
            matches = sum(1 for l in sample_lines if re.search(pattern, l))
            if matches > len(sample_lines) * 0.5:
                structure["timestamp_format"] = name
                break
        
        # Detect log levels
        level_counts = {}
        level_pattern = r'\b(DEBUG|INFO|WARN(?:ING)?|ERROR|FATAL|CRITICAL|TRACE)\b'
        for line in sample_lines:
            match = re.search(level_pattern, line, re.IGNORECASE)
            if match:
                level = match.group(1).upper()
                if level == "WARNING":
                    level = "WARN"
                level_counts[level] = level_counts.get(level, 0) + 1
        
        if level_counts:
            structure["log_levels"] = level_counts
        
        return FileProfile(
            path=rel_path,
            file_type="log",
            structure=structure,
            read_params={"encoding": encoding},
            row_count=total_lines,
            sample="\n".join(lines[:min(15, len(lines))]),
            content_summary=f"Log file with {total_lines} lines"
                           + (f", levels: {level_counts}" if level_counts else ""),
        )
    
    # =========================================================================
    # SQLite Profiler
    # =========================================================================
    
    def _profile_sqlite(self, full_path: str, rel_path: str) -> FileProfile:
        """Profile SQLite database: tables, schemas, relationships, samples."""
        try:
            conn = sqlite3.connect(f"file:{full_path}?mode=ro", uri=True)
        except Exception as e:
            return FileProfile(path=rel_path, file_type="sqlite",
                               error=f"Failed to open database: {str(e)[:200]}")
        
        try:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            table_names = [row[0] for row in cursor.fetchall()]
            
            tables = []
            total_rows = 0
            
            for table_name in table_names:
                # Column info
                cursor.execute(f"PRAGMA table_info('{table_name}')")
                columns = []
                for row in cursor.fetchall():
                    columns.append({
                        "cid": row[0],
                        "name": row[1],
                        "type": row[2],
                        "notnull": bool(row[3]),
                        "default": row[4],
                        "pk": bool(row[5]),
                    })
                
                # Row count
                cursor.execute(f"SELECT COUNT(*) FROM '{table_name}'")
                row_count = cursor.fetchone()[0]
                total_rows += row_count
                
                # Sample rows
                cursor.execute(f"SELECT * FROM '{table_name}' LIMIT {self.max_sample_rows}")
                sample_rows = cursor.fetchall()
                col_names = [c["name"] for c in columns]
                
                table_info = {
                    "name": table_name,
                    "columns": columns,
                    "row_count": row_count,
                    "sample_rows": [dict(zip(col_names, row)) for row in sample_rows],
                }
                tables.append(table_info)
            
            # Foreign keys
            foreign_keys = []
            for table_name in table_names:
                cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
                for row in cursor.fetchall():
                    foreign_keys.append({
                        "from_table": table_name,
                        "from_column": row[3],
                        "to_table": row[2],
                        "to_column": row[4],
                    })
            
            # Indexes
            indexes = []
            for table_name in table_names:
                cursor.execute(f"PRAGMA index_list('{table_name}')")
                for row in cursor.fetchall():
                    indexes.append({
                        "table": table_name,
                        "name": row[1],
                        "unique": bool(row[2]),
                    })
            
            structure = {
                "tables": tables,
                "foreign_keys": foreign_keys,
                "indexes": indexes,
                "table_count": len(tables),
            }
            
            # Build sample text (limit columns for wide tables to avoid huge markdown)
            sample_parts = []
            for t in tables[:3]:
                sample_parts.append(f"Table: {t['name']} ({t['row_count']} rows)")
                if t["sample_rows"]:
                    import pandas as pd
                    try:
                        df = pd.DataFrame(t["sample_rows"])
                        sample_parts.append(self._sample_df_for_markdown(df).to_markdown(index=False))
                    except Exception:
                        sample_parts.append(str(t["sample_rows"][:3]))
            sample = "\n\n".join(sample_parts)
            
            conn.close()
            
            return FileProfile(
                path=rel_path,
                file_type="sqlite",
                structure=structure,
                read_params={},
                row_count=total_rows,
                sample=sample[:3000],
                content_summary=f"SQLite database with {len(tables)} table(s), "
                               f"{total_rows} total rows",
            )
        
        except Exception as e:
            conn.close()
            return FileProfile(path=rel_path, file_type="sqlite",
                               error=f"Profiling failed: {str(e)[:200]}")
    
    # =========================================================================
    # NumPy Profiler
    # =========================================================================
    
    def _profile_numpy(self, full_path: str, rel_path: str) -> FileProfile:
        """Profile NPY/NPZ file: shape, dtype, basic statistics."""
        np = _try_import("numpy")
        if np is None:
            return FileProfile(path=rel_path, file_type="numpy",
                               error="numpy not installed")
        
        try:
            if full_path.endswith('.npz'):
                data = np.load(full_path, allow_pickle=False)
                arrays = {}
                for key in data.files:
                    arr = data[key]
                    arrays[key] = {
                        "shape": list(arr.shape),
                        "dtype": str(arr.dtype),
                    }
                    if np.issubdtype(arr.dtype, np.number):
                        arrays[key]["stats"] = {
                            "min": float(np.nanmin(arr)),
                            "max": float(np.nanmax(arr)),
                            "mean": float(np.nanmean(arr)),
                            "std": float(np.nanstd(arr)),
                        }
                
                structure = {
                    "format": "npz",
                    "arrays": arrays,
                    "array_count": len(arrays),
                }
                row_count = sum(a["shape"][0] for a in arrays.values() if a["shape"])
                
                sample = json.dumps(
                    {k: {**v, "first_values": data[k].flat[:5].tolist()}
                     for k, v in list(arrays.items())[:3]},
                    indent=2, default=str
                )[:2000]
                
            else:  # .npy
                arr = np.load(full_path, allow_pickle=False)
                structure = {
                    "format": "npy",
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                }
                if np.issubdtype(arr.dtype, np.number):
                    structure["stats"] = {
                        "min": float(np.nanmin(arr)),
                        "max": float(np.nanmax(arr)),
                        "mean": float(np.nanmean(arr)),
                        "std": float(np.nanstd(arr)),
                    }
                row_count = arr.shape[0] if arr.ndim > 0 else 1
                sample = str(arr.flat[:20].tolist())
            
            return FileProfile(
                path=rel_path,
                file_type="numpy",
                structure=structure,
                read_params={},
                row_count=row_count,
                sample=sample,
                content_summary=f"NumPy file, shape: {structure.get('shape', 'multiple arrays')}",
            )
        
        except Exception as e:
            return FileProfile(path=rel_path, file_type="numpy",
                               error=f"Failed to load: {str(e)[:200]}")
    
    # =========================================================================
    # NetCDF Profiler
    # =========================================================================
    
    def _profile_netcdf(self, full_path: str, rel_path: str) -> FileProfile:
        """Profile NetCDF file: dimensions, variables, attributes."""
        xr = _try_import("xarray")
        if xr is None:
            return FileProfile(path=rel_path, file_type="netcdf",
                               error="xarray not installed")
        
        try:
            ds = xr.open_dataset(full_path)
            
            structure = {
                "dimensions": {name: size for name, size in ds.dims.items()},
                "variables": [],
                "coordinates": list(ds.coords.keys()),
                "global_attributes": dict(ds.attrs),
            }
            
            for var_name in ds.data_vars:
                var = ds[var_name]
                var_info = {
                    "name": var_name,
                    "dims": list(var.dims),
                    "shape": list(var.shape),
                    "dtype": str(var.dtype),
                    "attributes": dict(var.attrs),
                }
                structure["variables"].append(var_info)
            
            ds.close()
            
            return FileProfile(
                path=rel_path,
                file_type="netcdf",
                structure=structure,
                read_params={},
                row_count=None,
                sample=str(structure["variables"][:3])[:2000],
                content_summary=f"NetCDF file with {len(structure['variables'])} variable(s), "
                               f"dims: {structure['dimensions']}",
            )
        except Exception as e:
            return FileProfile(path=rel_path, file_type="netcdf",
                               error=f"xarray failed: {str(e)[:200]}")
    
    # =========================================================================
    # Geo Profiler
    # =========================================================================
    
    def _profile_geo(self, full_path: str, rel_path: str) -> FileProfile:
        """
        Profile GeoPackage/Shapefile: layers, CRS, bbox, feature count,
        property types, attribute sample data, geometry details.
        """
        gpd = _try_import("geopandas")
        if gpd is not None:
            try:
                return self._profile_geo_geopandas(full_path, rel_path, gpd)
            except Exception:
                pass
        
        fiona = _try_import("fiona")
        if fiona is None:
            return FileProfile(path=rel_path, file_type="geo",
                               error="Neither geopandas nor fiona installed")
        
        try:
            structure = {"layers": []}
            total_features = 0
            
            layers = fiona.listlayers(full_path)
            for layer_name in layers:
                with fiona.open(full_path, layer=layer_name) as src:
                    props = src.schema.get("properties", {})
                    layer_info = {
                        "name": layer_name,
                        "feature_count": len(src),
                        "geometry_type": src.schema.get("geometry", "unknown"),
                        "properties": list(props.keys()),
                        "property_types": dict(props),
                        "crs": str(src.crs) if src.crs else None,
                        "bounds": list(src.bounds) if src.bounds else None,
                    }
                    sample_records = []
                    for i, feat in enumerate(src):
                        if i >= self.max_sample_rows:
                            break
                        sample_records.append(dict(feat.get("properties", {})))
                    layer_info["sample_records"] = sample_records
                    structure["layers"].append(layer_info)
                    total_features += len(src)
            
            sample_text = ""
            if structure["layers"]:
                layer0 = structure["layers"][0]
                if layer0.get("sample_records"):
                    import pandas as pd
                    sample_df = pd.DataFrame(layer0["sample_records"])
                    sample_text = self._sample_df_for_markdown(sample_df).to_markdown(index=False)
                    if len(sample_text) > 2000:
                        sample_text = sample_text[:2000] + "\n..."
                else:
                    sample_text = str(structure["layers"][:2])[:2000]
            
            return FileProfile(
                path=rel_path,
                file_type="geo",
                structure=structure,
                read_params={},
                row_count=total_features,
                sample=sample_text,
                content_summary=f"Geo file: {len(structure['layers'])} layer(s), "
                               f"{total_features} features",
            )
        except Exception as e:
            return FileProfile(path=rel_path, file_type="geo",
                               error=f"Profiling failed: {str(e)[:200]}")
    
    def _profile_geo_geopandas(self, full_path: str, rel_path: str, gpd) -> FileProfile:
        """Detailed geo profiling via geopandas."""
        import pandas as pd
        
        fiona = _try_import("fiona")
        layer_names = fiona.listlayers(full_path) if fiona is not None else [None]
        
        structure = {"layers": []}
        total_features = 0
        combined_sample = ""
        
        for layer_name in layer_names:
            kwargs = {"layer": layer_name} if layer_name else {}
            gdf = gpd.read_file(full_path, **kwargs)
            
            geom_col = gdf.geometry.name if hasattr(gdf, 'geometry') and gdf.geometry is not None else None
            geom_types = list(gdf.geometry.geom_type.unique()) if geom_col else []
            
            attr_cols = [c for c in gdf.columns if c != geom_col]
            attr_df = gdf[attr_cols] if attr_cols else pd.DataFrame()
            
            dtypes = {col: str(dtype) for col, dtype in attr_df.dtypes.items()} if len(attr_df) > 0 else {}
            numeric_cols = list(attr_df.select_dtypes(include=['number']).columns)
            categorical_cols = list(attr_df.select_dtypes(include=['object', 'category']).columns)
            missing = {col: int(attr_df[col].isna().sum()) for col in attr_cols if attr_df[col].isna().any()}
            
            crs_str = str(gdf.crs) if gdf.crs is not None else None
            crs_name = getattr(gdf.crs, 'name', None) if gdf.crs is not None else None
            bounds = list(gdf.total_bounds) if len(gdf) > 0 else None
            
            layer_info = {
                "name": layer_name or "default",
                "feature_count": len(gdf),
                "geometry_types": geom_types,
                "properties": attr_cols,
                "property_types": dtypes,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "missing_values": missing,
                "crs": crs_str,
                "crs_name": crs_name,
                "bounds": bounds,
            }
            structure["layers"].append(layer_info)
            total_features += len(gdf)
            
            if not combined_sample and len(attr_df) > 0:
                sample_df = self._sample_df_for_markdown(attr_df)
                if geom_col and len(gdf) > 0:
                    sample_df = sample_df.copy()
                    sample_df.insert(0, "_geom_type", gdf.geometry.geom_type.head(self.max_sample_rows).values)
                combined_sample = sample_df.to_markdown(index=False)
                if len(combined_sample) > 2000:
                    combined_sample = combined_sample[:2000] + "\n..."
        
        if structure["layers"]:
            layer0 = structure["layers"][0]
            structure["columns"] = layer0.get("properties", [])
            structure["numeric_columns"] = layer0.get("numeric_columns", [])
            structure["missing_values"] = layer0.get("missing_values", {})
        
        return FileProfile(
            path=rel_path,
            file_type="geo",
            structure=structure,
            read_params={},
            row_count=total_features,
            sample=combined_sample,
            content_summary=f"Geo file: {len(structure['layers'])} layer(s), "
                           f"{total_features} features, CRS: {structure['layers'][0].get('crs_name', 'unknown')}",
        )
    
    # =========================================================================
    # Parquet Profiler
    # =========================================================================
    
    def _profile_parquet(self, full_path: str, rel_path: str) -> FileProfile:
        """
        Profile Apache Parquet file: schema, row groups, column statistics,
        compression codec, and sample data.
        """
        import pandas as pd
        pq = _try_import("pyarrow.parquet")
        
        if pq is not None:
            try:
                pf = pq.ParquetFile(full_path)
                metadata = pf.metadata
                schema = pf.schema_arrow
                
                columns = [schema.field(i).name for i in range(len(schema))]
                dtypes = {schema.field(i).name: str(schema.field(i).type)
                          for i in range(len(schema))}
                numeric_arrow = {"int8", "int16", "int32", "int64",
                                 "uint8", "uint16", "uint32", "uint64",
                                 "float", "float16", "float32", "float64", "double"}
                numeric_cols = [c for c, dt in dtypes.items()
                                if any(nt in dt.lower() for nt in numeric_arrow)]
                
                row_groups = []
                for i in range(min(metadata.num_row_groups, 5)):
                    rg = metadata.row_group(i)
                    row_groups.append({
                        "num_rows": rg.num_rows,
                        "total_byte_size": rg.total_byte_size,
                    })
                
                column_stats = {}
                if metadata.num_row_groups > 0:
                    rg0 = metadata.row_group(0)
                    for j in range(rg0.num_columns):
                        col_chunk = rg0.column(j)
                        col_name = columns[j] if j < len(columns) else f"col_{j}"
                        stats = {}
                        if col_chunk.compression is not None:
                            stats["compression"] = str(col_chunk.compression)
                        if col_chunk.is_stats_set:
                            cs = col_chunk.statistics
                            if cs.has_min_max:
                                stats["min"] = str(cs.min)
                                stats["max"] = str(cs.max)
                            if cs.has_null_count:
                                stats["null_count"] = cs.null_count
                            if cs.has_distinct_count and cs.distinct_count > 0:
                                stats["distinct_count"] = cs.distinct_count
                        if stats:
                            column_stats[col_name] = stats
                
                missing = {}
                for col, st in column_stats.items():
                    nc = st.get("null_count", 0)
                    if nc and nc > 0:
                        missing[col] = nc
                
                try:
                    ncols = min(len(columns), self.max_sample_columns)
                    tbl = pf.read_row_group(0, columns=columns[:ncols])
                    sample_df = self._sample_df_for_markdown(tbl.to_pandas())
                    sample_text = sample_df.to_markdown(index=False)
                    if len(sample_text) > 2000:
                        sample_text = sample_text[:2000] + "\n..."
                except Exception:
                    sample_text = f"Columns: {columns[:20]}"
                
                structure = {
                    "columns": columns,
                    "num_columns": len(columns),
                    "num_rows": metadata.num_rows,
                    "dtypes": dtypes,
                    "numeric_columns": numeric_cols,
                    "missing_values": missing,
                    "num_row_groups": metadata.num_row_groups,
                    "row_groups": row_groups,
                    "column_stats": column_stats,
                    "created_by": metadata.created_by or None,
                    "format_version": metadata.format_version or None,
                }
                self._compute_all_column_patterns(structure)
                
                return FileProfile(
                    path=rel_path,
                    file_type="parquet",
                    structure=structure,
                    read_params={"engine": "pyarrow"},
                    row_count=metadata.num_rows,
                    sample=sample_text,
                    content_summary=f"Parquet: {metadata.num_rows} rows × {len(columns)} cols, "
                                   f"{metadata.num_row_groups} row group(s)",
                )
            except Exception:
                pass
        
        try:
            df = pd.read_parquet(full_path)
            columns = list(df.columns)
            dtypes = {col: str(dt) for col, dt in df.dtypes.items()}
            numeric_cols = list(df.select_dtypes(include=["number"]).columns)
            categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)
            missing = {col: int(df[col].isna().sum()) for col in columns if df[col].isna().any()}
            
            sample_text = self._sample_df_for_markdown(df).to_markdown(index=False)
            if len(sample_text) > 2000:
                sample_text = sample_text[:2000] + "\n..."
            
            structure = {
                "columns": columns,
                "num_columns": len(columns),
                "num_rows": len(df),
                "dtypes": dtypes,
                "numeric_columns": numeric_cols,
                "categorical_columns": categorical_cols,
                "missing_values": missing,
            }
            self._compute_all_column_patterns(structure)
            
            return FileProfile(
                path=rel_path,
                file_type="parquet",
                structure=structure,
                read_params={"engine": "pandas"},
                row_count=len(df),
                sample=sample_text,
                content_summary=f"Parquet: {len(df)} rows × {len(columns)} cols",
                dataframe=df,
            )
        except Exception as e:
            return FileProfile(path=rel_path, file_type="parquet",
                               error=f"Profiling failed: {str(e)[:200]}")
    
    # =========================================================================
    # dBASE (DBF) Profiler
    # =========================================================================
    
    def _profile_dbf(self, full_path: str, rel_path: str) -> FileProfile:
        """
        Profile dBASE (.dbf) file: field definitions (name, type, length,
        decimal precision), record count, encoding, and sample records.
        """
        import pandas as pd
        
        dbfread = _try_import("dbfread")
        if dbfread is not None:
            try:
                table = dbfread.DBF(full_path, load=False)
                columns = [f.name for f in table.fields]
                type_labels = {"C": "string", "N": "numeric", "F": "float",
                               "D": "date", "L": "boolean", "M": "memo"}
                dtypes = {f.name: type_labels.get(f.type, f.type) for f in table.fields}
                field_details = {
                    f.name: {
                        "type": f.type,
                        "type_label": type_labels.get(f.type, f.type),
                        "length": f.length,
                        "decimal_count": f.decimal_count,
                    }
                    for f in table.fields
                }
                numeric_cols = [f.name for f in table.fields if f.type in ("N", "F")]
                
                records = []
                for i, record in enumerate(table):
                    if i >= self.max_sample_rows:
                        break
                    records.append(dict(record))
                record_count = len(table)
                
                missing = {}
                if records:
                    sample_df = pd.DataFrame(records)
                    missing = {col: int(sample_df[col].isna().sum())
                               for col in sample_df.columns if sample_df[col].isna().any()}
                    sample_text = self._sample_df_for_markdown(sample_df).to_markdown(index=False)
                    if len(sample_text) > 2000:
                        sample_text = sample_text[:2000] + "\n..."
                else:
                    sample_text = f"Columns: {columns}"
                
                structure = {
                    "columns": columns,
                    "num_columns": len(columns),
                    "num_rows": record_count,
                    "dtypes": dtypes,
                    "field_details": field_details,
                    "numeric_columns": numeric_cols,
                    "missing_values": missing,
                    "encoding": getattr(table, "encoding", None),
                }
                self._compute_all_column_patterns(structure)
                
                return FileProfile(
                    path=rel_path,
                    file_type="dbf",
                    structure=structure,
                    read_params={"encoding": getattr(table, "encoding", None)},
                    row_count=record_count,
                    sample=sample_text,
                    content_summary=f"dBASE: {record_count} records × {len(columns)} fields",
                )
            except Exception:
                pass
        
        gpd = _try_import("geopandas")
        if gpd is not None:
            try:
                df = gpd.read_file(full_path)
                if "geometry" in df.columns:
                    df = df.drop(columns="geometry", errors="ignore")
                return self._dbf_from_dataframe(df, rel_path)
            except Exception:
                pass
        
        simpledbf = _try_import("simpledbf")
        if simpledbf is not None:
            try:
                dbf = simpledbf.Dbf5(full_path)
                return self._dbf_from_dataframe(dbf.to_dataframe(), rel_path)
            except Exception:
                pass
        
        return FileProfile(path=rel_path, file_type="dbf",
                           error="No DBF reader available (install dbfread, geopandas, or simpledbf)")
    
    def _dbf_from_dataframe(self, df, rel_path: str) -> FileProfile:
        """Helper: build DBF profile from a pandas DataFrame."""
        columns = list(df.columns)
        dtypes = {col: str(dt) for col, dt in df.dtypes.items()}
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        missing = {col: int(df[col].isna().sum()) for col in columns if df[col].isna().any()}
        sample_text = self._sample_df_for_markdown(df).to_markdown(index=False)
        if len(sample_text) > 2000:
            sample_text = sample_text[:2000] + "\n..."
        structure = {"columns": columns, "num_columns": len(columns),
                     "num_rows": len(df), "dtypes": dtypes,
                     "numeric_columns": numeric_cols, "missing_values": missing}
        self._compute_all_column_patterns(structure)
        return FileProfile(
            path=rel_path, file_type="dbf",
            structure=structure,
            row_count=len(df), sample=sample_text,
            content_summary=f"dBASE: {len(df)} records × {len(columns)} fields",
        )
    
    # =========================================================================
    # Shapefile Companion Profiler (.shx, .prj, .cpg)
    # =========================================================================
    
    def _profile_geo_companion(self, full_path: str, rel_path: str) -> FileProfile:
        """
        Profile shapefile companion files:
        - .prj: CRS / projection definition (WKT text)
        - .cpg: code page / encoding declaration
        - .shx: spatial index (binary — extract record count from header)
        """
        ext = self._get_extension(full_path).lower()
        associated_shp = os.path.splitext(rel_path)[0] + ".shp"
        
        try:
            file_size = os.path.getsize(full_path)
            
            if ext == ".prj":
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    wkt = f.read().strip()
                crs_name = None
                pyproj = _try_import("pyproj")
                if pyproj is not None:
                    try:
                        crs_name = pyproj.CRS.from_wkt(wkt).name
                    except Exception:
                        pass
                if crs_name is None:
                    import re as _re
                    m = _re.match(r'(?:PROJCS|GEOGCS)\["([^"]+)"', wkt)
                    if m:
                        crs_name = m.group(1)
                return FileProfile(
                    path=rel_path, file_type="geo_companion",
                    structure={"companion_type": "projection", "wkt": wkt[:2000],
                               "crs_name": crs_name, "associated_shp": associated_shp},
                    sample=wkt[:500],
                    content_summary=f"Projection file (CRS: {crs_name or 'see WKT'})",
                )
            
            elif ext == ".cpg":
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    encoding = f.read().strip()
                return FileProfile(
                    path=rel_path, file_type="geo_companion",
                    structure={"companion_type": "codepage", "encoding": encoding,
                               "associated_shp": associated_shp},
                    sample=encoding,
                    content_summary=f"Code page file (encoding: {encoding})",
                )
            
            elif ext == ".shx":
                record_count = None
                try:
                    import struct
                    with open(full_path, "rb") as f:
                        header = f.read(100)
                    if len(header) >= 100:
                        file_len_words = struct.unpack(">i", header[24:28])[0]
                        record_count = (file_len_words - 50) // 4
                except Exception:
                    pass
                return FileProfile(
                    path=rel_path, file_type="geo_companion",
                    structure={"companion_type": "spatial_index", "file_size_bytes": file_size,
                               "record_count": record_count, "associated_shp": associated_shp},
                    row_count=record_count,
                    sample=f"Spatial index: {file_size} bytes"
                           + (f", {record_count} records" if record_count else ""),
                    content_summary=f"Spatial index ({record_count or '?'} records, {file_size} bytes)",
                )
            
            else:
                return FileProfile(
                    path=rel_path, file_type="geo_companion",
                    structure={"companion_type": ext, "file_size_bytes": file_size,
                               "associated_shp": associated_shp},
                    sample=f"Companion file ({ext}), {file_size} bytes",
                    content_summary=f"Geo companion ({ext})",
                )
        except Exception as e:
            return FileProfile(path=rel_path, file_type="geo_companion",
                               error=f"Profiling failed: {str(e)[:200]}")
    
    # =========================================================================
    # Compressed File Profiler
    # =========================================================================
    
    def _profile_compressed(self, full_path: str, rel_path: str) -> FileProfile:
        """
        Handle compressed files: decompress to temp dir, profile contents,
        report inner file profiles.
        """
        ext = self._get_extension(full_path)
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                inner_files = []
                
                if ext == '.gz':
                    # GZ usually wraps a single file
                    inner_name = os.path.basename(full_path)
                    if inner_name.endswith('.gz'):
                        inner_name = inner_name[:-3]
                    inner_path = os.path.join(tmpdir, inner_name)
                    
                    with gzip.open(full_path, 'rb') as f_in:
                        with open(inner_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    inner_files.append(inner_path)
                    
                elif ext == '.zip':
                    with zipfile.ZipFile(full_path, 'r') as zf:
                        zf.extractall(tmpdir)
                    # List extracted files
                    for root, dirs, fnames in os.walk(tmpdir):
                        for fname in fnames:
                            inner_files.append(os.path.join(root, fname))
                
                elif ext == '.tar':
                    with tarfile.open(full_path, 'r') as tf:
                        tf.extractall(tmpdir)
                    for root, dirs, fnames in os.walk(tmpdir):
                        for fname in fnames:
                            inner_files.append(os.path.join(root, fname))
                
                elif ext == '.tgz':
                    with tarfile.open(full_path, 'r:gz') as tf:
                        tf.extractall(tmpdir)
                    for root, dirs, fnames in os.walk(tmpdir):
                        for fname in fnames:
                            inner_files.append(os.path.join(root, fname))
                
                else:
                    return FileProfile(path=rel_path, file_type="compressed",
                                       error=f"Unsupported compression: {ext}")
                
                # Profile inner files
                inner_profiles = []
                for inner_path in inner_files[:20]:  # Limit to 20 files
                    inner_rel = os.path.relpath(inner_path, tmpdir)
                    inner_ext = self._get_extension(inner_path)
                    if inner_ext in EXTENSION_MAP:
                        p = self.profile_file(inner_path, f"{rel_path}/{inner_rel}")
                        inner_profiles.append(p)
                
                if inner_profiles:
                    structure = {
                        "compression": ext.lstrip('.'),
                        "inner_file_count": len(inner_files),
                        "inner_profiles": [
                            {"path": p.path, "type": p.file_type, "rows": p.row_count}
                            for p in inner_profiles
                        ],
                    }
                    
                    # Return the first inner profile as the main one if only one file
                    if len(inner_profiles) == 1:
                        p = inner_profiles[0]
                        p.path = rel_path  # Use compressed file path
                        p.content_summary = f"[Compressed] {p.content_summary}"
                        return p
                    
                    return FileProfile(
                        path=rel_path,
                        file_type="compressed",
                        structure=structure,
                        row_count=None,
                        content_summary=f"Compressed archive with {len(inner_files)} file(s)",
                    )
                else:
                    return FileProfile(
                        path=rel_path, file_type="compressed",
                        structure={"compression": ext.lstrip('.'),
                                   "inner_file_count": len(inner_files)},
                        content_summary=f"Compressed archive with {len(inner_files)} file(s) "
                                       f"(no supported file types found inside)",
                    )
        
        except Exception as e:
            return FileProfile(path=rel_path, file_type="compressed",
                               error=f"Decompression failed: {str(e)[:200]}")
    
    # =========================================================================
    # Column Pattern Detection
    # =========================================================================
    
    @staticmethod
    def _detect_column_patterns(columns: List[str], min_group_size: int = 5) -> dict:
        """
        Detect repeating naming patterns in column names.
        Groups columns like gene_0, gene_1, ..., gene_20531 into a single pattern.
        
        Args:
            columns: List of column names
            min_group_size: Minimum columns to form a pattern group
            
        Returns:
            dict with:
            - "patterns": list of {"pattern": str, "count": int, "range": [first, last]}
            - "standalone": list of column names that don't match any pattern
            - "summary": human-readable string like "['id'] + gene_{N} (20531 cols)"
        """
        # Group by pattern: replace numeric sequences with {N}
        pattern_buckets = defaultdict(list)
        for col in columns:
            pattern = re.sub(r'\d+', '{N}', col)
            pattern_buckets[pattern].append(col)
        
        patterns = []
        standalone = []
        
        for pattern, members in pattern_buckets.items():
            if len(members) >= min_group_size and pattern != members[0]:
                # It's a real pattern (has numeric parts that vary)
                patterns.append({
                    "pattern": pattern,
                    "count": len(members),
                    "range": [members[0], members[-1]],
                })
            else:
                standalone.extend(members)
        
        # Build summary string
        parts = []
        if standalone:
            parts.append(str(standalone[:80]))
            if len(standalone) > 80:
                parts[-1] = str(standalone[:80])[:-1] + f", ... +{len(standalone)-80} more]"
        for p in patterns:
            parts.append(f"{p['pattern']} ({p['count']} cols, {p['range'][0]} ~ {p['range'][1]})")
        
        return {
            "patterns": patterns,
            "standalone": standalone,
            "summary": " + ".join(parts) if parts else str(columns[:80]),
        }
    
    @staticmethod
    def _compute_all_column_patterns(structure: dict) -> None:
        """Pre-compute column_patterns, numeric_column_patterns, categorical_column_patterns in-place."""
        cols = structure.get("columns", [])
        if cols and "column_patterns" not in structure:
            structure["column_patterns"] = AutoProfiler._detect_column_patterns(cols)
        num_cols = structure.get("numeric_columns", [])
        if num_cols:
            structure["numeric_column_patterns"] = AutoProfiler._detect_column_patterns(num_cols)
        cat_cols = structure.get("categorical_columns", [])
        if cat_cols:
            structure["categorical_column_patterns"] = AutoProfiler._detect_column_patterns(cat_cols)
    
    # =========================================================================
    # Utilities
    # =========================================================================
    
    @staticmethod
    def _get_extension(file_path: str) -> str:
        """Get file extension, handling double extensions like .tar.gz"""
        name = os.path.basename(file_path).lower()
        if name.endswith('.tar.gz'):
            return '.tgz'
        _, ext = os.path.splitext(name)
        return ext
