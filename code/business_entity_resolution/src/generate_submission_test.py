import pandas as pd
import lightgbm as lgb
import sqlite3
import re
import csv
from pathlib import Path
from rapidfuzz import fuzz

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR.parent.parent / "dataset" / "test"
OUTPUT_DIR = PROJECT_DIR / "output"

SRC1 = DATA_DIR / "test_source1.tsv"
SRC2 = DATA_DIR / "test_source2.tsv"
SRC3 = DATA_DIR / "test_source3.tsv"
CANDIDATES = OUTPUT_DIR / "candidate_pairs.tsv"
MODEL_PATH = OUTPUT_DIR / "matcher_model.txt"
DB_PATH = OUTPUT_DIR / "entities_test.sqlite"   # separate DB for test data
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


def build_db():
    print("Building test entity database...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS records")
    cur.execute("""
        CREATE TABLE records (
            entity_id TEXT PRIMARY KEY,
            name TEXT,
            address TEXT,
            country TEXT
        )
    """)

    for path in (SRC1, SRC2, SRC3):
        print(f"  loading {path.name}...")
        n = 0
        batch = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                name = normalize(row.get("business_name", ""))
                addr = normalize(row.get("business_address", ""))
                country = (row.get("country") or "").strip().lower()
                batch.append((row["entity_id"], name, addr, country))
                n += 1
                if len(batch) >= 5000:
                    cur.executemany("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?)", batch)
                    batch = []
                if n % 1_000_000 == 0:
                    print(f"    {n:,} rows")
        if batch:
            cur.executemany("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?)", batch)

    conn.commit()
    conn.close()
    print("Test database built.")


if not DB_PATH.exists():
    build_db()
else:
    print("Using existing test entity database.")

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


all_s1_ids = []
with open(SRC1, encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        all_s1_ids.append(row["entity_id"])

print(f"Total test S1 entities: {len(all_s1_ids):,}")

results = {}

print("Scoring candidates and generating final matches...")

with open(CANDIDATES, encoding="utf-8", newline="") as in_f:
    reader = csv.DictReader(in_f, delimiter="\t")
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
            s1_name_tok = set(s1_name.split())
            s1_addr_tok = set(s1_addr.split())

            feats, valid_cands = [], []
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
    writer = csv.writer(out_f, delimiter="\t")
    writer.writerow(["source1_entity_id", "matched_entity_ids"])
    for s1_id in all_s1_ids:
        matched = results.get(s1_id, [])
        writer.writerow([s1_id, ",".join(matched)])

print(f"Done -> {RESULTS_OUT}")
print(f"Entities with at least one match: {sum(1 for v in results.values() if v):,}")