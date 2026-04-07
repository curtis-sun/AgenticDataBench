import re
from collections import defaultdict
from tqdm import tqdm
from nltk.stem import PorterStemmer

def normalize_word(word: str) -> str:
    # Remove non-alphanumeric (keep internal apostrophes if needed, but usually not)
    clean = re.sub(r"[^a-z0-9']", '', word)
    if not clean:
        return word
    stemmer = PorterStemmer()
    clean = stemmer.stem(clean)
    return clean

def tokenize_and_normalize(s: str) -> list:
    # Split on separators (same as your original)
    tokens = re.split(r'[\s\-_/&]+', s.strip().lower())
    words = []
    for token in tokens:
        norm = normalize_word(token)
        if norm:
            words.append(norm)
    return words

def get_word_set(s: str) -> frozenset:
    return frozenset(tokenize_and_normalize(s))

def get_wordset_cached(s, wordset_cache: dict) -> frozenset:
        if s not in wordset_cache:
            wordset_cache[s] = get_word_set(s)
        return wordset_cache[s]

def check_name_covers(concepts: set[str], threshold: float) -> tuple[list, dict]:
    concept_to_canonical = {}
    word_to_concepts = defaultdict(set)  # word → set of current canonicals
    wordset_cache = {}

    def overlap_coefficient(set_a, set_b):
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / min(len(set_a), len(set_b))

    for concept in tqdm(concepts, desc="Clustering"):
        words = get_wordset_cached(concept, wordset_cache)
        if not words:
            concept_to_canonical[concept] = concept
            continue

        # Find candidate canonicals that share any word
        candidate_canons = set()
        for w in words:
            candidate_canons.update(word_to_concepts[w])

        best_canon = None
        best_sim = 0.0
        best_canon_length = None

        for canon in candidate_canons:
            canon_words = get_wordset_cached(canon, wordset_cache)
            sim = overlap_coefficient(words, canon_words)
            if sim >= threshold and sim > best_sim:
                best_sim = sim
                best_canon = canon
                best_canon_length = len(canon_words)

        if best_canon is not None:
            # Choose shorter as canonical
            if len(words) < best_canon_length:
                new_canon = concept

                # Remap all concepts pointing to old canonical
                for orig, rep in list(concept_to_canonical.items()):
                    if rep == best_canon:
                        concept_to_canonical[orig] = new_canon

                # Remove old canonical from word index
                old_words = get_wordset_cached(best_canon, wordset_cache)
                for w in old_words:
                    word_to_concepts[w].discard(best_canon)

                # Add new canonical to index
                for w in words:
                    word_to_concepts[w].add(new_canon)

                concept_to_canonical[concept] = new_canon
            else:
                concept_to_canonical[concept] = best_canon
                # Optional: boost recall by indexing its words too
                for w in words:
                    word_to_concepts[w].add(best_canon)
        else:
            # New cluster
            concept_to_canonical[concept] = concept
            for w in words:
                word_to_concepts[w].add(concept)

    canonical_to_originals = defaultdict(list)
    for orig in concepts:
        canon = concept_to_canonical.get(orig, orig)
        canonical_to_originals[canon].append(orig)

    sorted_clusters = sorted(canonical_to_originals.items(), key=lambda x: -len(x[1]))
    return sorted_clusters, wordset_cache