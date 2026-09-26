import pandas as pd
import lightgbm as lgb
import sqlite3
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR.parent.parent / "dataset" / "train"
OUTPUT_DIR = PROJECT_DIR / "output"

SRC1 = DATA_DIR / "train_source1.tsv"
CANDIDATES = OUTPUT_DIR / "candidate_pairs.tsv"
MODEL_PATH = OUTPUT_DIR / "matcher_model.txt"
DB_PATH = OUTPUT_DIR / "entities.sqlite"
RESULTS_OUT = OUTPUT_DIR / "matching_results.tsv"

THRESHOLD = 0.7
FEATURE_COLS = [
    "name_jaccard", "name_fuzz_ratio", "name_token_sort_ratio",
    "addr_jaccard", "addr_fuzz_ratio", "country_match",
]

TOKEN_RE = re.compile(r"[^a-z0-9\s]")


def normalize(text):
    if not text:
        return ""
    text = TOKEN_RE.sub(" ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


from rapidfuzz import fuzz

print("Loading model...")
model = lgb.Booster(model_file=str(MODEL_PATH))

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

BATCH_SIZE = 500


def fetch_records(entity_ids):
    result = {}
    ids = list(entity_ids)
    for i in range(0, len(ids), BATCH_SIZE):
        chunk = ids[i:i + BATCH_SIZE]
        placeholders = ",".join("?" * len(chunk))
        cur.execute(
            f"SELECT entity_id, name, address, country FROM records WHERE entity_id IN ({placeholders})",
            chunk,
        )
        for eid, name, addr, country in cur.fetchall():
            result[eid] = (name, addr, country)
    return result


# every S1 entity must appear in the output, even with zero candidates
all_s1_ids = []
with open(SRC1, encoding="utf-8", newline="") as f:
    import csv
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        all_s1_ids.append(row["entity_id"])

print(f"Total S1 entities: {len(all_s1_ids):,}")

results = {}  # s1_id -> list of matched candidate ids

print("Scoring candidates and generating final matches...")

import csv as csv_module

with open(CANDIDATES, encoding="utf-8", newline="") as in_f:
    reader = csv_module.DictReader(in_f, delimiter="\t")

    rows_batch = []
    BATCH_ROWS = 2000
    n = 0

    def process_batch(rows_batch):
        needed_ids = set()
        for s1_id, cand_ids in rows_batch:
            needed_ids.add(s1_id)
            needed_ids.update(cand_ids)

        records = fetch_records(needed_ids)

        for s1_id, cand_ids in rows_batch:
            if s1_id not in records:
                results[s1_id] = []
                continue
            s1_name, s1_addr, s1_country = records[s1_id]
            s1_name = normalize(s1_name) if s1_name and " " not in s1_name.strip() else s1_name  # already normalized in DB
            s1_name_tok = set(s1_name.split())
            s1_addr_tok = set(s1_addr.split())

            feats = []
            valid_cands = []
            for cand_id in cand_ids:
                if cand_id not in records:
                    continue
                c_name, c_addr, c_country = records[cand_id]
                c_name_tok = set(c_name.split())
                c_addr_tok = set(c_addr.split())

                name_jac = jaccard(s1_name_tok, c_name_tok)
                name_fuzz = fuzz.ratio(s1_name, c_name) / 100.0
                name_sort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0
                addr_jac = jaccard(s1_addr_tok, c_addr_tok)
                addr_fuzz = fuzz.ratio(s1_addr, c_addr) / 100.0
                country_match = 1 if s1_country and s1_country == c_country else 0

                feats.append([name_jac, name_fuzz, name_sort, addr_jac, addr_fuzz, country_match])
                valid_cands.append(cand_id)

            if feats:
                X = pd.DataFrame(feats, columns=FEATURE_COLS)
                probs = model.predict(X)
                matched = [cid for cid, p in zip(valid_cands, probs) if p >= THRESHOLD]
                results[s1_id] = matched
            else:
                results[s1_id] = []

    for row in reader:
        cand_ids = [c.strip() for c in row["candidate_entity_ids"].split(",") if c.strip()]
        rows_batch.append((row["source1_entity_id"], cand_ids))
        n += 1

        if len(rows_batch) >= BATCH_ROWS:
            process_batch(rows_batch)
            rows_batch = []

        if n % 200_000 == 0:
            print(f"  processed {n:,} S1 entities")

    if rows_batch:
        process_batch(rows_batch)

conn.close()

print("Writing matching_results.tsv...")
with open(RESULTS_OUT, "w", encoding="utf-8", newline="") as out_f:
    writer = csv_module.writer(out_f, delimiter="\t")
    writer.writerow(["source1_entity_id", "matched_entity_ids"])
    for s1_id in all_s1_ids:
        matched = results.get(s1_id, [])
        writer.writerow([s1_id, ",".join(matched)])

print(f"Done -> {RESULTS_OUT}")
print(f"Entities with at least one match: {sum(1 for v in results.values() if v):,}")