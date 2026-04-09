"""
Dataset Loader (Multi-Type)

Supports loading and profiling all file types:
- Structured: CSV, TSV, Excel
- Markup: JSON, YAML, HTML, Markdown, TeX
- Text: TXT, LOG
- Database: SQLite
- Binary: NPY, NPZ, NetCDF, GeoPackage
- Compressed: GZ, ZIP
"""

import os
from typing import List, Dict, Optional, Any, Set

from data_classes import DatasetFile, DomainDataset, FileProfile
from auto_profiler import AutoProfiler
import sys
sys.path.append('..')
from utils.config import DATASETS_DIR

# ==================== Helper Functions ====================

def _get_pattern_summary(patterns: dict, columns: list = None) -> str:
    """
    Extract summary from pre-computed column patterns dict.
    Falls back to str(columns) if patterns is None or has no summary.
    """
    if patterns and patterns.get('summary'):
        return patterns['summary']
    if columns is not None:
        return str(columns)
    return "[]"


def format_file_info_line(f: dict) -> str:
    """Format a single file_info dict into a one-line summary."""
    ft = f.get('file_type', 'tabular')
    name = f['name']
    if ft in ('csv', 'excel', 'tabular', 'parquet', 'dbf'):
        col_desc = _get_pattern_summary(f.get('column_patterns'), f.get('columns', [])[:8])
        return f"- {name}: {ft}, {f.get('num_rows', '?')} rows, {f.get('num_columns', '?')} cols, columns: {col_desc}"
    elif ft in ('json', 'yaml'):
        return f"- {name}: {ft}, {f.get('row_count', '?')} records, keys: {f.get('top_level_keys', [])[:8]}"
    elif ft == 'sqlite':
        return f"- {name}: {ft}, tables: {[t['name'] for t in f.get('tables', [])]}"
    elif ft in ('text', 'log'):
        return f"- {name}: {ft}, {f.get('line_count', '?')} lines"
    elif ft == 'geo_companion':
        return f"- {name}: {f.get('companion_type', '?')} companion for {f.get('associated_shp', '?')}"
    else:
        return f"- {name}: {ft}"

    
def _get_file_group_key(name: str) -> str:
    """
    Derive a grouping key from a file name/path.
    Files with the same key likely share internal structure and columns.
    Groups by: subfolder + filename prefix (all parts except the last underscore-separated token).
    E.g., 'Zillow House Price Data/City_MedianRentalPrice_1Bedroom.csv'
        -> 'Zillow House Price Data/City_MedianRentalPrice'
    """
    dirname = os.path.dirname(name)
    basename = os.path.splitext(os.path.basename(name))[0]
    parts = basename.split('_')
    if len(parts) > 1:
        prefix = '_'.join(parts[:-1])
    else:
        prefix = basename
    return f"{dirname}/{prefix}" if dirname else prefix


def build_files_summary(
    files_info: List[Dict],
    detailed: bool = False,
    max_files: Optional[int] = None,
    include_samples: bool = True,
) -> str:
    """
    Build files summary string for prompts.
    
    Updated to handle both tabular and non-tabular file profiles.
    Backward compatible: still works with old-style files_info dicts.
    
    When max_files is set and len(files_info) > max_files, files are grouped by
    naming pattern: one representative per group gets full detail (incl. sample_data),
    other files in the group are listed as "same structure". This keeps all file names
    visible to the LLM while dramatically reducing token count.
    
    If include_samples is False, sample_data blocks are omitted (saves significant tokens).
    """
    total_files = len(files_info)

    # When there are many files, group by naming pattern: show full detail for
    # one representative per group, list the rest as "same structure".
    file_to_representative = {}  # non-rep file name -> representative name
    group_others = {}            # representative name -> [other file names in group]
    if max_files is not None and total_files > max_files:
        groups = {}
        for f in files_info:
            key = _get_file_group_key(f['name'])
            groups.setdefault(key, []).append(f)
        for key, group_files in groups.items():
            rep_name = group_files[0]['name']
            if len(group_files) > 1:
                others = [f['name'] for f in group_files[1:]]
                group_others[rep_name] = others
                for other_name in others:
                    file_to_representative[other_name] = rep_name

    summary = ""
    for f in files_info:
        name = f['name']

        # Skip non-representative files (they are listed under their representative)
        if name in file_to_representative:
            continue
        file_type = f.get("file_type", "tabular")
        summary += f"\n### File: {f['name']}\n"
        
        if file_type in ("csv", "excel", "tabular", "parquet", "dbf"):
            # Tabular format (backward compatible)
            if detailed:
                summary += f"- Type: {file_type}\n"
                summary += f"- Rows: {f.get('num_rows', '?')}, Columns: {f.get('num_columns', '?')}\n"
            
            cols = f.get('columns', [])
            if cols:
                summary += f"- Columns: {_get_pattern_summary(f.get('column_patterns'), cols)}\n"
            
            num_cols = f.get('numeric_columns', [])
            if num_cols:
                summary += f"- Numeric: {_get_pattern_summary(f.get('numeric_column_patterns'), num_cols)}\n"
            if detailed:
                cat_cols = f.get('categorical_columns', [])
                if cat_cols:
                    summary += f"- Categorical: {_get_pattern_summary(f.get('categorical_column_patterns'), cat_cols)}\n"
                missing = f.get('missing_values', {})
                if missing and any(v > 0 for v in missing.values()):
                    missing_nonzero = {k: v for k, v in missing.items() if v > 0}
                    missing_keys = list(missing_nonzero.keys())
                    pattern_result = AutoProfiler._detect_column_patterns(missing_keys)
                    if pattern_result['patterns']:
                        summary += f"- Missing values ({len(missing_nonzero)} cols): {pattern_result['summary']}\n"
                    else:
                        summary += f"- Missing values: {missing_nonzero}\n"
                # Parquet-specific: column stats from row group metadata
                col_stats = f.get('column_stats', {})
                if col_stats:
                    summary += f"- Column stats (row group 0): {dict(list(col_stats.items())[:5])}\n"
                # DBF-specific: field definitions with type/length/decimal
                field_details = f.get('field_details', {})
                if field_details:
                    summary += f"- Field details: {dict(list(field_details.items())[:8])}\n"
            if include_samples and 'sample_data' in f and f['sample_data']:
                summary += f"- Sample:\n{f['sample_data']}\n"
        
        elif file_type in ("json", "yaml"):
            summary += f"- Type: {file_type}\n"
            top_type = f.get("top_level_type", "")
            if top_type:
                summary += f"- Structure: {top_type}\n"
            keys = f.get("top_level_keys", [])
            if keys:
                summary += f"- Keys: {keys}\n"
            row_count = f.get("row_count")
            if row_count:
                summary += f"- Records: {row_count}\n"
            if include_samples and 'sample_data' in f and f['sample_data']:
                summary += f"- Sample:\n{f['sample_data']}\n"
        
        elif file_type == "sqlite":
            summary += f"- Type: SQLite database\n"
            tables = f.get("tables", [])
            for t in tables:
                summary += (f"  - Table `{t['name']}`: {t.get('row_count', '?')} rows, "
                           f"columns: {[c['name'] for c in t.get('columns', [])]}\n")
            fks = f.get("foreign_keys", [])
            if fks:
                summary += "- Foreign keys:\n"
                for fk in fks:
                    summary += (f"  - `{fk['from_table']}`.`{fk['from_column']}` → "
                               f"`{fk['to_table']}`.`{fk['to_column']}`\n")
            if include_samples and 'sample_data' in f and f['sample_data']:
                summary += f"- Sample:\n{f['sample_data']}\n"
        
        elif file_type in ("numpy", "netcdf"):
            summary += f"- Type: {file_type}\n"
            shape = f.get("shape")
            if shape:
                summary += f"- Shape: {shape}\n"
            variables = f.get("variables")
            if variables:
                summary += f"- Variables: {variables}\n"
            if include_samples and 'sample_data' in f and f['sample_data']:
                summary += f"- Sample:\n{f['sample_data']}\n"
        
        elif file_type == "geo":
            summary += f"- Type: geospatial\n"
            layers = f.get("layers", [])
            if layers:
                for layer in layers[:3]:
                    summary += f"  - Layer '{layer.get('name', '?')}': {layer.get('feature_count', '?')} features"
                    geom = layer.get("geometry_types") or [layer.get("geometry_type")]
                    if geom:
                        summary += f", geometry: {geom}"
                    summary += "\n"
                    props = layer.get("properties", [])
                    if props:
                        summary += f"    Properties: {props[:10]}\n"
                    crs = layer.get("crs_name") or layer.get("crs")
                    if crs:
                        summary += f"    CRS: {crs}\n"
            cols = f.get("columns", [])
            if cols:
                summary += f"- Columns: {cols}\n"
            if include_samples and 'sample_data' in f and f['sample_data']:
                summary += f"- Sample:\n{f['sample_data']}\n"
        
        elif file_type == "geo_companion":
            companion = f.get("companion_type", "unknown")
            assoc = f.get("associated_shp", "")
            summary += f"- Type: {companion} companion for {assoc}\n"
            crs_name = f.get("crs_name")
            if crs_name:
                summary += f"- CRS: {crs_name}\n"
            encoding = f.get("encoding")
            if encoding:
                summary += f"- Encoding: {encoding}\n"
            rec_count = f.get("record_count")
            if rec_count:
                summary += f"- Records: {rec_count}\n"
        
        elif file_type in ("html", "markdown", "tex"):
            summary += f"- Type: {file_type}\n"
            headings = f.get("headings", [])
            if headings:
                summary += f"- Headings: {headings[:5]}\n"
            tables_count = f.get("tables_count", 0)
            if tables_count:
                summary += f"- Tables: {tables_count}\n"
            elements = f.get("elements", {})
            if elements:
                summary += f"- Elements: {elements}\n"
            if include_samples and 'sample_data' in f and f['sample_data']:
                summary += f"- Sample:\n{f['sample_data'][:500]}\n"
        
        elif file_type in ("text", "log"):
            summary += f"- Type: {file_type}\n"
            line_count = f.get("line_count")
            if line_count:
                summary += f"- Lines: {line_count}\n"
            log_levels = f.get("log_levels")
            if log_levels:
                summary += f"- Log levels: {log_levels}\n"
            if include_samples and 'sample_data' in f and f['sample_data']:
                summary += f"- Sample:\n{f['sample_data'][:500]}\n"
        
        else:
            summary += f"- Type: {file_type}\n"
            if include_samples and 'sample_data' in f and f['sample_data']:
                summary += f"- Sample:\n{f['sample_data'][:500]}\n"

        # After a representative file's detail, list other files in its group
        if name in group_others:
            others = group_others[name]
            summary += f"Files with same structure (use any of them): {', '.join(others)}\n"

    return summary


def _profile_to_file_info(profile: FileProfile) -> List[Dict[str, Any]]:
    """
    Convert a FileProfile to a list of files_info dicts.
    
    For multi-sheet Excel, returns one dict per sheet.
    For all other types, returns a single-element list.
    """
    if profile.error:
        return [{"name": profile.path, "file_type": profile.file_type, "error": profile.error}]
    
    if profile.file_type == "excel":
        s = profile.structure
        sheets = s.get("sheets", [])
        if len(sheets) > 1:
            # Multi-sheet: one entry per sheet
            results = []
            for sheet in sheets:
                info = {
                    "name": f"{profile.path} ({sheet['name']})",
                    "file_type": "excel",
                    "num_rows": sheet.get("num_rows", 0),
                    "num_columns": sheet.get("num_columns", 0),
                    "columns": sheet.get("columns", []),
                    "dtypes": sheet.get("dtypes", {}),
                    "missing_values": sheet.get("missing_values", {}),
                    "numeric_columns": sheet.get("numeric_columns", []),
                    "categorical_columns": sheet.get("categorical_columns", []),
                    "column_patterns": sheet.get("column_patterns"),
                    "numeric_column_patterns": sheet.get("numeric_column_patterns"),
                    "categorical_column_patterns": sheet.get("categorical_column_patterns"),
                    "sample_data": profile.sample if sheet == sheets[0] else None,
                }
                results.append(info)
            return results
        elif sheets:
            # Single sheet
            sheet = sheets[0]
            return [{
                "name": profile.path,
                "file_type": "excel",
                "num_rows": sheet.get("num_rows", 0),
                "num_columns": sheet.get("num_columns", 0),
                "columns": sheet.get("columns", []),
                "dtypes": sheet.get("dtypes", {}),
                "missing_values": sheet.get("missing_values", {}),
                "numeric_columns": sheet.get("numeric_columns", []),
                "categorical_columns": sheet.get("categorical_columns", []),
                "column_patterns": sheet.get("column_patterns"),
                "numeric_column_patterns": sheet.get("numeric_column_patterns"),
                "categorical_column_patterns": sheet.get("categorical_column_patterns"),
                "sample_data": profile.sample,
            }]
    
    # Everything below returns a single-element list — same as before but wrapped in []
    info = {
        "name": profile.path,
        "file_type": profile.file_type,
    }
    
    if profile.file_type == "csv":
        s = profile.structure
        info.update({
            "num_rows": s.get("num_rows", profile.row_count or 0),
            "num_columns": s.get("num_columns", 0),
            "columns": s.get("columns", []),
            "dtypes": s.get("dtypes", {}),
            "missing_values": s.get("missing_values", {}),
            "numeric_columns": s.get("numeric_columns", []),
            "categorical_columns": s.get("categorical_columns", []),
            "column_patterns": s.get("column_patterns"),
            "numeric_column_patterns": s.get("numeric_column_patterns"),
            "categorical_column_patterns": s.get("categorical_column_patterns"),
            "sample_data": profile.sample,
        })
    
    elif profile.file_type in ("json", "yaml"):
        s = profile.structure
        info.update({
            "top_level_type": s.get("top_level_type", ""),
            "top_level_keys": s.get("top_level_keys", []),
            "depth": s.get("depth", 0),
            "record_count": s.get("record_count"),
            "row_count": profile.row_count,
            "sample_data": profile.sample,
        })
        if "schema" in s:
            info["schema"] = s["schema"]
    
    elif profile.file_type == "sqlite":
        s = profile.structure
        info.update({
            "tables": s.get("tables", []),
            "foreign_keys": s.get("foreign_keys", []),
            "indexes": s.get("indexes", []),
            "row_count": profile.row_count,
            "sample_data": profile.sample,
        })
    
    elif profile.file_type in ("numpy", "netcdf"):
        s = profile.structure
        info.update({
            "shape": s.get("shape"),
            "dtype": s.get("dtype"),
            "variables": [v.get("name") for v in s.get("variables", [])],
            "dimensions": s.get("dimensions"),
            "row_count": profile.row_count,
            "sample_data": profile.sample,
        })
    
    elif profile.file_type == "geo":
        s = profile.structure
        info.update({
            "layers": s.get("layers"),
            "columns": s.get("columns", []),
            "numeric_columns": s.get("numeric_columns", []),
            "missing_values": s.get("missing_values", {}),
            "row_count": profile.row_count,
            "sample_data": profile.sample,
        })
    
    elif profile.file_type == "parquet":
        s = profile.structure
        info.update({
            "num_rows": s.get("num_rows", profile.row_count or 0),
            "num_columns": s.get("num_columns", 0),
            "columns": s.get("columns", []),
            "dtypes": s.get("dtypes", {}),
            "missing_values": s.get("missing_values", {}),
            "numeric_columns": s.get("numeric_columns", []),
            "categorical_columns": s.get("categorical_columns", []),
            "column_patterns": s.get("column_patterns"),
            "numeric_column_patterns": s.get("numeric_column_patterns"),
            "categorical_column_patterns": s.get("categorical_column_patterns"),
            "num_row_groups": s.get("num_row_groups"),
            "column_stats": s.get("column_stats", {}),
            "sample_data": profile.sample,
        })
    
    elif profile.file_type == "dbf":
        s = profile.structure
        info.update({
            "num_rows": s.get("num_rows", profile.row_count or 0),
            "num_columns": s.get("num_columns", 0),
            "columns": s.get("columns", []),
            "dtypes": s.get("dtypes", {}),
            "field_details": s.get("field_details", {}),
            "missing_values": s.get("missing_values", {}),
            "numeric_columns": s.get("numeric_columns", []),
            "column_patterns": s.get("column_patterns"),
            "numeric_column_patterns": s.get("numeric_column_patterns"),
            "encoding": s.get("encoding"),
            "sample_data": profile.sample,
        })
    
    elif profile.file_type == "geo_companion":
        s = profile.structure
        info.update({
            "companion_type": s.get("companion_type"),
            "associated_shp": s.get("associated_shp"),
            "crs_name": s.get("crs_name"),
            "encoding": s.get("encoding"),
            "record_count": s.get("record_count"),
            "wkt": s.get("wkt", "")[:200],
            "sample_data": profile.sample,
        })
    
    elif profile.file_type in ("html", "markdown", "tex"):
        s = profile.structure
        info.update({
            "headings": s.get("headings", []),
            "elements": s.get("elements", {}),
            "tables_count": s.get("tables_count", 0),
            "line_count": s.get("line_count"),
            # If HTML had tables, include tabular info too
            "columns": s.get("columns", []),
            "numeric_columns": s.get("numeric_columns", []),
            "row_count": profile.row_count,
            "sample_data": profile.sample,
        })
    
    elif profile.file_type in ("text", "log"):
        s = profile.structure
        info.update({
            "line_count": s.get("line_count"),
            "is_log": s.get("is_log", False),
            "log_levels": s.get("log_levels"),
            "timestamp_format": s.get("timestamp_format"),
            "detected_format": s.get("detected_format"),
            "row_count": profile.row_count,
            "sample_data": profile.sample,
        })
    
    else:
        info.update({
            "row_count": profile.row_count,
            "sample_data": profile.sample,
        })
    
    return [info]


def explore_domain_dataset(domain_dataset: DomainDataset) -> Dict[str, Any]:
    """
    Explore all files in a domain and return combined characteristics.
    
    Updated to work with multi-type files via FileProfile.
    Backward compatible: still produces the same top-level dict structure.
    """
    exploration = {
        "domain": domain_dataset.domain,
        "num_files": len(domain_dataset.files),
        "total_rows": 0,
        "has_missing_values": False,
        "all_columns": [],
        "numeric_columns": [],
        "files_info": [],
        # New fields for multi-type support
        "file_types": {},       # count of each file type
        "has_non_tabular": False,
    }
    
    all_columns = set()
    numeric_columns = set()
    type_counts = {}
    
    for file in domain_dataset.files:
        profile = file.profile
        
        if profile is not None:
            # New path: use FileProfile
            file_infos = _profile_to_file_info(profile)
            exploration["files_info"].extend(file_infos)
            
            # Track file types
            ft = profile.file_type
            type_counts[ft] = type_counts.get(ft, 0) + 1
            if ft not in ("csv", "excel", "parquet", "dbf"):
                exploration["has_non_tabular"] = True
            
            # Aggregate tabular info
            if profile.row_count:
                exploration["total_rows"] += profile.row_count
            
            cols = profile.structure.get("columns", [])
            all_columns.update(cols)
            numeric_columns.update(profile.structure.get("numeric_columns", []))
            
            missing = profile.structure.get("missing_values", {})
            if missing and any(v > 0 for v in missing.values()):
                exploration["has_missing_values"] = True
            
    exploration["all_columns"] = list(all_columns)
    exploration["numeric_columns"] = list(numeric_columns)
    exploration["file_types"] = type_counts
    
    # Group files by subfolder.  Each subfolder is treated as one logical
    # file group.  Within each subfolder we also detect shared columns
    # (potential join keys).  Companion files (.dbf that duplicates a .shp,
    # geo_companion types) are excluded from the shared-column computation
    # so the result is clean.
    subfolder_profiles: Dict[str, List[FileProfile]] = {}
    for file in domain_dataset.files:
        profile = file.profile
        if profile and not profile.error:
            parts = profile.path.replace("\\", "/").split("/")
            subfolder = "/".join(parts[:-1]) if len(parts) > 1 else ""
            subfolder_profiles.setdefault(subfolder, []).append(profile)

    # Pre-compute set of .shp base names so we can skip redundant .dbf files
    shp_bases: Set[str] = set()
    for file in domain_dataset.files:
        if file.profile and file.profile.file_type == "geo":
            shp_bases.add(os.path.splitext(file.profile.path)[0])

    subfolders = {}
    for folder, profiles_in_folder in subfolder_profiles.items():
        file_basenames = [os.path.basename(p.path) for p in profiles_in_folder]

        # Detect shared columns within this subfolder (skip companion / redundant files)
        col_files: Dict[str, List[str]] = {}
        for p in profiles_in_folder:
            if p.file_type == "geo_companion":
                continue
            if p.file_type == "dbf" and os.path.splitext(p.path)[0] in shp_bases:
                continue
            for col in p.structure.get("columns", []):
                col_files.setdefault(col, []).append(os.path.basename(p.path))
        shared_columns = {
            col: fnames for col, fnames in col_files.items() if len(fnames) >= 2
        }

        subfolders[folder] = {
            "files": file_basenames,
            "shared_columns": shared_columns,
        }
    exploration["subfolders"] = subfolders

    return exploration


# ==================== Dataset Loader (Multi-Type) ====================

METADATA_FILES = {"domain_connections.txt"}


class DatasetLoader:
    """
    Load and profile datasets from directory.
    
    Supports all file types via AutoProfiler.
    Backward compatible: load_domain() still returns DomainDataset with
    DatasetFile objects. Tabular files have .dataframe populated.
    """
    
    def __init__(self, datasets_dir: str = DATASETS_DIR, llm_client=None):
        self.datasets_dir = datasets_dir
        self.profiler = AutoProfiler(llm_client=llm_client)
    
    def get_all_domains(self) -> List[str]:
        """Get all domain names"""
        if not os.path.exists(self.datasets_dir):
            return []
        return sorted([d for d in os.listdir(self.datasets_dir)
                if os.path.isdir(os.path.join(self.datasets_dir, d))])
    
    def load_domain(self, domain: str) -> Optional[DomainDataset]:
        """
        Load all files in a domain with auto-profiling.
        
        Returns DomainDataset with:
        - files: List[DatasetFile] with .profile (FileProfile) populated
        
        Tabular files (.csv, .xlsx) also have .dataframe populated for
        backward compatibility.
        """
        domain_path = os.path.join(self.datasets_dir, domain)
        if not os.path.exists(domain_path):
            print(f"Domain path not found: {domain_path}")
            return None
        
        # Profile all files (excluding internal metadata files)
        profiles = [
            p for p in self.profiler.profile_directory(domain_path)
            if os.path.basename(p.path) not in METADATA_FILES
        ]
        
        if not profiles:
            print(f"No supported files found in domain: {domain}")
            return None
        
        # Convert profiles to DatasetFile objects
        files = []
        for profile in profiles:
            name = os.path.splitext(profile.path)[0].replace(os.sep, '_')
            
            dataset_file = DatasetFile(
                name=name,
                file_path=os.path.join(domain_path, profile.path),
                profile=profile,
            )
            files.append(dataset_file)
        
        if not files:
            print(f"Failed to profile any files in domain: {domain}")
            return None
        
        return DomainDataset(
            domain=domain,
            files=files,
        )
    
