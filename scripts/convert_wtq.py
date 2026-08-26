"""
Convert WikiTableQuestions pristine-unseen split into ATR-compatible inputs.

Reads the pristine-unseen tsv and writes, under --out_dir:
  wtq_excel/wtq_<dir>_<id>.xlsx       one Excel per table
  wtq_unseen_flat.json                flat per-question records

Usage:
  python scripts/convert_wtq.py \
      --wtq_dir /path/to/WikiTableQuestions --out_dir $DATA_DIR
"""
import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd

DEFAULT_WTQ_DIR = Path("dataset/WikiTableQuestions")

# Bound at startup by main(); see _bind_paths.
WTQ_BASE = TSV = OUT_EXCEL = OUT_FLAT = None


def _bind_paths(wtq_dir: Path, out_dir: Path) -> None:
    global WTQ_BASE, TSV, OUT_EXCEL, OUT_FLAT
    WTQ_BASE = wtq_dir
    TSV = wtq_dir / "data/pristine-unseen-tables.tsv"
    OUT_EXCEL = out_dir / "wtq_excel"
    OUT_FLAT = out_dir / "wtq_unseen_flat.json"
    OUT_EXCEL.mkdir(parents=True, exist_ok=True)

def context_to_table_id(context: str) -> str:
    # 'csv/203-csv/733.csv' -> 'wtq_203_733'
    m = re.match(r"csv/(\d+)-csv/(\d+)\.csv", context)
    if not m:
        raise ValueError(f"unexpected context: {context}")
    return f"wtq_{m.group(1)}_{m.group(2)}"

def sanitize_columns(cols):
    seen = {}
    out = []
    for i, c in enumerate(cols):
        s = str(c).replace("\n", " ").strip()
        s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_") or f"col_{i}"
        if s[0].isdigit():
            s = f"c_{s}"
        if s in seen:
            seen[s] += 1
            out.append(f"{s}_{seen[s]}")
        else:
            seen[s] = 0
            out.append(s)
    return out

def load_csv(path: Path) -> pd.DataFrame:
    # Some WTQ CSVs have ragged rows; read raw, pad to max width.
    raw = list(csv.reader(open(path, newline="", encoding="utf-8")))
    if not raw:
        return pd.DataFrame()
    width = max(len(r) for r in raw)
    header = list(raw[0]) + [f"col_{i}" for i in range(len(raw[0]), width)]
    cols = sanitize_columns(header)
    data = [list(r) + [""] * (width - len(r)) for r in raw[1:]]
    return pd.DataFrame(data, columns=cols)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--wtq_dir", type=Path, default=DEFAULT_WTQ_DIR,
                    help="WikiTableQuestions checkout (default: %(default)s)")
    ap.add_argument("--out_dir", type=Path, default=Path("data"),
                    help="directory to write the converted split into "
                         "(default: %(default)s)")
    args = ap.parse_args()
    tsv = args.wtq_dir / "data/pristine-unseen-tables.tsv"
    if not tsv.exists():
        raise SystemExit(f"WTQ split not found: {tsv}")
    _bind_paths(args.wtq_dir, args.out_dir)

    rows = list(csv.reader(open(TSV), delimiter="\t"))
    rows = rows[1:]                       # drop the tsv header row
    print(f"loaded {len(rows)} questions from {TSV.name}")

    contexts = sorted({r[2] for r in rows})
    print(f"unique tables: {len(contexts)}")

    saved_tables = 0
    skipped_tables = 0
    for ctx in contexts:
        table_id = context_to_table_id(ctx)
        csv_path = WTQ_BASE / ctx
        try:
            df = load_csv(csv_path)
        except Exception as exc:
            print(f"  load fail {ctx}: {exc}")
            skipped_tables += 1
            continue
        if df.empty or len(df.columns) == 0:
            skipped_tables += 1
            continue
        try:
            df.to_excel(OUT_EXCEL / f"{table_id}.xlsx", index=False)
            saved_tables += 1
        except Exception as exc:
            print(f"  save fail {table_id}: {exc}")
            skipped_tables += 1

    flat = []
    skipped_q = 0
    for r in rows:
        qid, utterance, ctx, target = r[0], r[1], r[2], r[3]
        try:
            table_id = context_to_table_id(ctx)
        except ValueError:
            skipped_q += 1
            continue
        # WTQ targets can be multi-value separated by '|'
        answer_text = target.replace("|", " | ")
        flat.append({
            "question_id": qid,
            "question": utterance,
            "table_id": table_id,
            "answer-text": answer_text,
            "_wtq": {"target_raw": target, "context": ctx},
        })

    OUT_FLAT.write_text(json.dumps(flat, ensure_ascii=False, indent=2))
    print(f"  excel saved: {saved_tables} (skipped {skipped_tables})")
    print(f"  flat records: {len(flat)} (skipped {skipped_q})")

if __name__ == "__main__":
    main()
