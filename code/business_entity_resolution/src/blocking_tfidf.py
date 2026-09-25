import csv
import re
import math
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR.parent.parent / "dataset" / "train"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

SRC1 = DATA_DIR / "train_source1.tsv"
SRC2 = DATA_DIR / "train_source2.tsv "
SRC3 = DATA_DIR / "train_source3.tsv"

TOP_K = 20                  # candidates kept per S1 entity
MAX_POSTING_LIST = 2000     # tokens appearing in more records than this are dropped (too common, no signal)
MAX_TOKENS_PER_ENTITY = 12  # only score using each entity's rarest N tokens (speed cap)
MIN_TOKEN_LEN = 2

STOPWORDS = {
    "the", "and", "of", "inc", "incorporated", "corp", "corporation", "ltd",
    "limited", "pvt", "private", "llc", "co", "company", "group", "holdings",
    "enterprises",
}

TOKEN_RE = re.compile(r"[^a-z0-9\s]")


def tokenize(text):
    if not text:
        return []
    text = TOKEN_RE.sub(" ", text.lower())
    return [t for t in text.split() if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS]


def iter_records(path):
    """Yield (entity_id, token_set) combining name + address. One row at a time."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name_tokens = tokenize(row.get("business_name", ""))
            addr_tokens = tokenize(row.get("business_address", ""))
            yield row["entity_id"], set(name_tokens + addr_tokens)


def run_blocking(pool_paths, query_path, output_path):
    # ---------- Pass 1: document frequency ----------
    print("Pass 1: counting token frequencies...")
    doc_freq = Counter()
    total_docs = 0

    for path in pool_paths:
        n = 0
        for _, tokens in iter_records(path):
            for t in tokens:
                doc_freq[t] += 1
            n += 1
            total_docs += 1
            if n % 500_000 == 0:
                print(f"  {path.name}: {n:,} rows")

    good_tokens = {t for t, c in doc_freq.items() if c <= MAX_POSTING_LIST}
    print(f"Distinct tokens: {len(doc_freq):,} | kept: {len(good_tokens):,}")

    idf = {t: math.log(total_docs / (1 + doc_freq[t])) for t in good_tokens}
    del doc_freq

    # ---------- Pass 2: inverted index ----------
    print("Pass 2: building inverted index...")
    inverted = defaultdict(list)

    for path in pool_paths:
        n = 0
        for entity_id, tokens in iter_records(path):
            for t in tokens:
                if t in good_tokens:
                    inverted[t].append(entity_id)
            n += 1
            if n % 500_000 == 0:
                print(f"  {path.name}: {n:,} rows")

    print(f"Inverted index built: {len(inverted):,} tokens")

    # ---------- Generate candidates ----------
    print("Generating candidates...")
    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        n = 0
        for entity_id, tokens in iter_records(query_path):
            valid_tokens = [t for t in tokens if t in inverted]

            if len(valid_tokens) > MAX_TOKENS_PER_ENTITY:
                valid_tokens = sorted(valid_tokens, key=lambda t: len(inverted[t]))[:MAX_TOKENS_PER_ENTITY]

            scores = defaultdict(float)
            for t in valid_tokens:
                w = idf.get(t, 0)
                for cand_id in inverted[t]:
                    scores[cand_id] += w

            if scores:
                top = sorted(scores.items(), key=lambda x: -x[1])[:TOP_K]
                cand_ids = ",".join(c for c, _ in top)
            else:
                cand_ids = ""

            writer.writerow([entity_id, cand_ids])
            n += 1
            if n % 200_000 == 0:
                print(f"  processed {n:,} S1 entities")

    print(f"Done -> {output_path}")


if __name__ == "__main__":
    run_blocking(
        pool_paths=[SRC2, SRC3],
        query_path=SRC1,
        output_path=OUTPUT_DIR / "candidate_pairs.tsv",
    )