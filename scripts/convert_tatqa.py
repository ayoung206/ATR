"""
Convert TAT-QA dev set into ATR-compatible inputs.

Reads the official TAT-QA dev json and writes, under --out_dir:
  tatqa_excel/tat_<short_uid>.xlsx       one Excel per table
  tatqa_doc/tat_<short_uid>.json         paragraphs as {pseudo-url: text}
  tatqa_dev_flat.json                    flat per-question records

Usage:
  python scripts/convert_tatqa.py \
      --src /path/to/tatqa_dataset_dev.json --out_dir $DATA_DIR
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

DEFAULT_SRC = Path("dataset/TAT-QA-master/dataset_raw/tatqa_dataset_dev.json")

# Bound at startup by main(); see _bind_paths.
SRC = OUT_EXCEL = OUT_DOC = OUT_FLAT = None


def _bind_paths(src: Path, out_dir: Path) -> None:
    global SRC, OUT_EXCEL, OUT_DOC, OUT_FLAT
    SRC = src
    OUT_EXCEL = out_dir / "tatqa_excel"
    OUT_DOC = out_dir / "tatqa_doc"
    OUT_FLAT = out_dir / "tatqa_dev_flat.json"
    OUT_EXCEL.mkdir(parents=True, exist_ok=True)
    OUT_DOC.mkdir(parents=True, exist_ok=True)

def _short_id(uid: str) -> str:
    return uid.replace("-", "")[:12]

def _normalize_table(rows):
    """Take 2D list, return DataFrame with single header row.

    TAT-QA tables sometimes have multi-row headers; we collapse the first
    1-2 rows into a single header. Heuristic: if the first row has many empty
    cells, merge it with the second row.
    """
    if not rows:
        return pd.DataFrame()
    if len(rows) == 1:
        return pd.DataFrame(columns=[f"col_{i}" for i in range(len(rows[0]))])

    first = rows[0]
    second = rows[1] if len(rows) > 1 else []
    empty_first = sum(1 for c in first if not str(c).strip())
    use_two_row = (empty_first > len(first) // 2) and second
    if use_two_row:
        header = []
        for a, b in zip(first, second):
            a, b = str(a).strip(), str(b).strip()
            joined = f"{a} {b}".strip() if a and b else (a or b)
            header.append(joined or "col")
        data_rows = rows[2:]
    else:
        header = [str(c).strip() or f"col_{i}" for i, c in enumerate(first)]
        data_rows = rows[1:]

    # Make headers unique
    seen = {}
    unique_header = []
    for h in header:
        if not h:
            h = "col"
        # MySQL-safe: alphanumeric + underscore
        h = re.sub(r"[^0-9a-zA-Z]+", "_", h).strip("_") or "col"
        if h in seen:
            seen[h] += 1
            unique_header.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            unique_header.append(h)

    # Pad rows to header length
    width = len(unique_header)
    fixed_rows = []
    for r in data_rows:
        r = list(r) + [""] * (width - len(r))
        fixed_rows.append(r[:width])

    return pd.DataFrame(fixed_rows, columns=unique_header)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="official TAT-QA dev json (default: %(default)s)")
    ap.add_argument("--out_dir", type=Path, default=Path("data"),
                    help="directory to write the converted split into "
                         "(default: %(default)s)")
    args = ap.parse_args()
    if not args.src.exists():
        raise SystemExit(f"TAT-QA source not found: {args.src}")
    _bind_paths(args.src, args.out_dir)

    data = json.loads(SRC.read_text())
    print(f"loaded {len(data)} entries from {SRC.name}")

    flat_records = []
    skipped = 0
    for entry in data:
        table_uid = entry["table"]["uid"]
        short = _short_id(table_uid)
        table_id = f"tat_{short}"
        rows = entry["table"]["table"]

        df = _normalize_table(rows)
        if df.empty:
            skipped += 1
            continue

        # Save Excel
        xlsx_path = OUT_EXCEL / f"{table_id}.xlsx"
        try:
            df.to_excel(xlsx_path, index=False)
        except Exception as exc:
            print(f"  ⚠ skip {table_id}: {exc}")
            skipped += 1
            continue

        # Save paragraphs as doc JSON
        doc_dict = {}
        for p in entry.get("paragraphs", []):
            key = f"/tatqa/{table_id}/p{p.get('order', 0)}/{p.get('uid', '')[:8]}"
            doc_dict[key] = p.get("text", "")
        doc_path = OUT_DOC / f"{table_id}.json"
        doc_path.write_text(json.dumps(doc_dict, ensure_ascii=False, indent=2))

        # Flatten questions
        for q in entry.get("questions", []):
            answer = q.get("answer", "")
            if isinstance(answer, list):
                answer_text = " | ".join(str(a) for a in answer)
            else:
                answer_text = str(answer)
            scale = q.get("scale", "") or ""
            if scale and scale not in answer_text:
                answer_text_with_scale = f"{answer_text} {scale}".strip()
            else:
                answer_text_with_scale = answer_text

            flat_records.append({
                "question_id": q["uid"],
                "question": q["question"],
                "table_id": table_id,
                "answer-text": answer_text_with_scale,
                "_tatqa": {
                    "answer_raw": answer,
                    "answer_type": q.get("answer_type", ""),
                    "answer_from": q.get("answer_from", ""),
                    "scale": scale,
                    "derivation": q.get("derivation", ""),
                    "table_uid": table_uid,
                },
            })

    OUT_FLAT.write_text(json.dumps(flat_records, ensure_ascii=False, indent=2))
    print(f"  excel files: {len(list(OUT_EXCEL.glob('*.xlsx')))}")
    print(f"  doc files:   {len(list(OUT_DOC.glob('*.json')))}")
    print(f"  flat records: {len(flat_records)}")
    print(f"  skipped tables: {skipped}")

if __name__ == "__main__":
    main()
