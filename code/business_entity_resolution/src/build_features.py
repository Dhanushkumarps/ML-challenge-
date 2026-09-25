import csv
import re
from pathlib import Path
from collections import defaultdict

try:
    from rapidfuzz import fuzz
except ImportError:
    raise SystemExit("Run: pip install rapidfuzz")

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR.parent.parent / "dataset" / "train"
OUTPUT_DIR = PROJECT_DIR / "output"

SRC1 = DATA_DIR / "train_source1.tsv"
SRC2 = DATA_DIR / "train_source2.tsv"
SRC3 = DATA_DIR / "train_source3.tsv"
GROUND_TRUTH = DATA_DIR / "train_ground_truth.tsv"
CANDIDATES = OUTPUT_DIR / "candidate_pairs.tsv"

FEATURES_OUT = OUTPUT_DIR / "features.csv"

TOKEN_RE = re.compile(r"[^a-z0-9\s]")


def normalize(text):
    if not text:
        return ""
    text = TOKEN_RE.sub(" ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def tokens(text):
    return set(normalize(text).split())


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def load_records(path):
    """entity_id -> (norm_name, name_tokens, norm_addr, addr_tokens, country)"""
    out = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = normalize(row.get("business_name", ""))
            addr = normalize(row.get("business_address", ""))
            out[row["entity_id"]] = (
                name, set(name.split()),
                addr, set(addr.split()),
                (row.get("country") or "").strip().lower(),
            )
    return out


print("Loading records...")
s1 = load_records(SRC1)
s2 = load_records(SRC2)
s3 = load_records(SRC3)
all_targets = {**s2, **s3}
del s2, s3
print(f"S1: {len(s1):,}  targets(S2+S3): {len(all_targets):,}")

print("Loading ground truth...")
gt_pairs = set()
with open(GROUND_TRUTH, encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        if row["matched_entity_ids"].strip():
            for m in row["matched_entity_ids"].split(","):
                m = m.strip()
                if m:
                    gt_pairs.add((row["source1_entity_id"], m))
print(f"Ground-truth positive pairs: {len(gt_pairs):,}")

print("Computing features for candidate pairs...")

with open(CANDIDATES, encoding="utf-8", newline="") as in_f, \
     open(FEATURES_OUT, "w", encoding="utf-8", newline="") as out_f:

    reader = csv.DictReader(in_f, delimiter="\t")
    writer = csv.writer(out_f)
    writer.writerow([
        "source1_entity_id", "candidate_entity_id",
        "name_jaccard", "name_fuzz_ratio", "name_token_sort_ratio",
        "addr_jaccard", "addr_fuzz_ratio",
        "country_match",
        "label",
    ])

    n = 0
    for row in reader:
        s1_id = row["source1_entity_id"]
        if s1_id not in s1 or not row["candidate_entity_ids"].strip():
            continue

        s1_name, s1_name_tok, s1_addr, s1_addr_tok, s1_country = s1[s1_id]

        for cand_id in row["candidate_entity_ids"].split(","):
            cand_id = cand_id.strip()
            if cand_id not in all_targets:
                continue

            c_name, c_name_tok, c_addr, c_addr_tok, c_country = all_targets[cand_id]

            name_jac = jaccard(s1_name_tok, c_name_tok)
            name_fuzz = fuzz.ratio(s1_name, c_name) / 100.0
            name_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
            addr_jac = jaccard(s1_addr_tok, c_addr_tok)
            addr_fuzz = fuzz.ratio(s1_addr, c_addr) / 100.0
            country_match = 1 if s1_country and s1_country == c_country else 0
            label = 1 if (s1_id, cand_id) in gt_pairs else 0

            writer.writerow([
                s1_id, cand_id,
                round(name_jac, 4), round(name_fuzz, 4), round(name_sort, 4),
                round(addr_jac, 4), round(addr_fuzz, 4),
                country_match, label,
            ])

        n += 1
        if n % 200_000 == 0:
            print(f"  processed {n:,} S1 entities")

print(f"Done -> {FEATURES_OUT}")