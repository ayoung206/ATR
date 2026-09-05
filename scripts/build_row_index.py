"""
Phase F-A: rebuild the row component of a current-format MultiviewIndex.

Strategy: load the existing index payload (which already has table chunks +
schemas + cells encoded), re-scan the excel dir to extract row-level dicts,
encode them with the same BGE-M3 embedder, and save a new variant of the
index with the additional `.row.faiss` and `row_entries` payload.

Skips re-encoding the doc/schema/cell views, only the new RowIndex is built.
Legacy indices are intentionally rejected by ``MultiviewIndex.load``; rebuild
all views with ``build_index.py`` instead of using this script to migrate one.

Usage:
  python scripts/build_row_index.py \\
    --source_index index/hybridqa_multiview \\
    --excel_dir ./dataset/HybridQA/dev_excel \\
    --bge_dir <BGE_DIR> \\
    --save_path index/hybridqa_multiview_v6 \\
    --per_table_quota 100 \\
    --budget 500000 \\
    --device cuda
"""
from __future__ import annotations
import argparse
import os
import shutil
import sys
import time


from atr.offline.multiview_index import (
    MultiviewIndex,
    RowIndex,
    _load_dataframe,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_index", required=True,
                    help="Existing MultiviewIndex save_path (without extensions).")
    ap.add_argument("--excel_dir", required=True,
                    help="Directory of source xlsx/csv tables to re-scan for rows.")
    ap.add_argument("--bge_dir", required=True,
                    help="Path to bge-m3 model dir.")
    ap.add_argument("--save_path", required=True,
                    help="Destination prefix for new index (with V6 RowIndex).")
    ap.add_argument("--budget", type=int, default=500_000)
    ap.add_argument("--per_table_quota", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # 1. Copy existing .doc/.schema/.cell faiss + meta to new save_path
    print(f"[1/4] Copying existing index → {args.save_path}.*")
    src_meta = args.source_index + ".meta.pkl"
    if not os.path.exists(src_meta):
        sys.exit(f"meta not found: {src_meta}")
    for ext in (".doc.faiss", ".schema.faiss", ".cell.faiss"):
        src = args.source_index + ext
        if os.path.exists(src):
            shutil.copy(src, args.save_path + ext)
    # meta will be re-saved with new row_entries below

    # 2. Load source index payload + embedder
    print("[2/4] Loading source MultiviewIndex...")
    t0 = time.time()
    idx = MultiviewIndex.load(
        save_path=args.source_index,
        bge_model_path=args.bge_dir,
        device=args.device,
        require_cuda=(args.device == "cuda"),
        enable_reranker=False,  # offline index construction only
    )
    print(f"      loaded in {time.time()-t0:.1f}s")
    print(f"      doc_chunks={len(idx.doc_retriever.chunks)} schema={len(idx.schema_index.entries)} cells={len(idx.cell_index.entries)}")

    # 3. Build RowIndex by re-scanning excel_dir
    print("[3/4] Scanning excel_dir for row-level extraction...")
    idx.row_index = RowIndex(
        idx.embedder,
        budget=args.budget,
        per_table_quota=args.per_table_quota,
    )
    n_tables = 0
    t0 = time.time()
    files = sorted(os.listdir(args.excel_dir))
    for fname in files:
        lower = fname.lower()
        if not (lower.endswith(".xlsx") or lower.endswith(".csv")):
            continue
        fpath = os.path.join(args.excel_dir, fname)
        df = _load_dataframe(fpath)
        if df is None or df.empty:
            continue
        table_id = os.path.splitext(fname)[0]
        idx.row_index.add_table(table_id, df)
        n_tables += 1
        if n_tables % 200 == 0:
            print(f"      scanned {n_tables} tables, entries={len(idx.row_index.entries)}")
    print(f"      scanned {n_tables} tables → {len(idx.row_index.entries)} row entries  ({time.time()-t0:.1f}s)")

    print(f"      building FAISS index for {len(idx.row_index.entries)} rows…")
    t0 = time.time()
    idx.row_index.build()
    print(f"      built in {time.time()-t0:.1f}s")

    # 4. Re-save full payload to new path
    print(f"[4/4] Saving augmented index → {args.save_path}.*")
    t0 = time.time()
    idx.save_path = args.save_path
    idx.save()
    print(f"      saved in {time.time()-t0:.1f}s")

    print("\nDone. Verify with:")
    print(f"  python -c \"from atr.offline.multiview_index import MultiviewIndex; "
          f"idx = MultiviewIndex.load('{args.save_path}', "
          f"bge_model_path='{args.bge_dir}', device='cuda', require_cuda=True, "
          f"enable_reranker=False); "
          f"print(len(idx.row_index.entries), 'rows')\"")

if __name__ == "__main__":
    main()
