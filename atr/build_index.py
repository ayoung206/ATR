"""
§3.2  Offline Phase CLI

Builds the 5-view Multiview Index from heterogeneous documents:
  View 1: Text Chunks
  View 2: Table Chunks + mapping f
  View 3: Schema Index
  View 4: Cell Index (budget-aware)
  View 5: Relational DB (external Flask service, no build needed)

Usage:
    python build_index.py \\
        --excel_dir  $DATA_DIR/hybridqa/dev_excel \\
        --doc_dir    $DATA_DIR/hybridqa/dev_doc \\
        --bge_dir    BAAI \\
        --save_path  ./index/multiview \\
        --budget     10000
"""
import argparse
import logging
import os
import time

from atr.clients.chat_utils import init_logger                    # noqa: E402
from atr.config import CELL_INDEX_BUDGET                          # noqa: E402
from atr.offline.multiview_index import (                 # noqa: E402
    DOCUMENT_CHUNK_OVERLAP,
    DOCUMENT_CHUNK_SIZE,
    MultiviewIndex,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Agentic TableRAG Multiview Index")
    parser.add_argument("--excel_dir", required=True,
                        help="Directory containing Excel/CSV table files")
    parser.add_argument("--doc_dir", required=True,
                        help="Directory containing JSON/text document files")
    parser.add_argument("--bge_dir", default="BAAI",
                        help="Directory containing bge-m3 model")
    parser.add_argument("--save_path", required=True,
                        help="Output path prefix (e.g. ./index/multiview)")
    parser.add_argument("--budget", type=int, default=CELL_INDEX_BUDGET,
                        help="Cell Index global budget B (§3.2 View 4). "
                             "safety cap on total entries across all tables")
    parser.add_argument("--per_table_quota", type=int, default=50,
                        help="Cell Index per-table quota: max (col,value) "
                             "pairs encoded per table. Mirrors T2024 "
                             "max_encode_cell semantics so every table "
                             "contributes equally (fixes 6.1%% global-cap "
                             "coverage problem).")
    parser.add_argument(
        "--row_budget",
        type=int,
        default=0,
        help="Row Index global cap; 0 indexes every row (paper default)",
    )
    parser.add_argument(
        "--row_per_table_quota",
        type=int,
        default=0,
        help="Row Index per-table cap; 0 indexes every row (paper default)",
    )
    parser.add_argument(
        "--max_row_chars",
        type=int,
        default=0,
        help="Maximum serialized row characters; 0 keeps full rows (paper default)",
    )
    parser.add_argument(
        "--document_chunk_size",
        type=int,
        default=DOCUMENT_CHUNK_SIZE,
        help="Document chunk size in BGE-M3 tokens (paper default: 512)",
    )
    parser.add_argument(
        "--document_chunk_overlap",
        type=int,
        default=DOCUMENT_CHUNK_OVERLAP,
        help="Document chunk overlap in BGE-M3 tokens (paper default: 64)",
    )
    parser.add_argument("--device", default="auto",
                        help="Device for embedding: auto|cpu|cuda|cuda:0")
    parser.add_argument("--require_cuda", action="store_true")
    args = parser.parse_args()
    for name in ("row_budget", "row_per_table_quota", "max_row_chars"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be 0 or a positive integer")

    init_logger(name="build_index", level=logging.INFO, log_file=None)
    logger = logging.getLogger("build_index")

    bge_model_path = os.path.join(args.bge_dir, "bge-m3")
    logger.info(f"Building MultiviewIndex: excel={args.excel_dir}, doc={args.doc_dir}")
    logger.info(
        f"BGE model: {bge_model_path}, budget={args.budget}, "
        f"per_table_quota={args.per_table_quota}, device={args.device}, "
        f"row_budget={args.row_budget or 'all'}, "
        f"row_per_table_quota={args.row_per_table_quota or 'all'}, "
        f"document_chunks={args.document_chunk_size}/"
        f"{args.document_chunk_overlap} tokens"
    )

    start = time.time()
    index = MultiviewIndex(
        excel_dir=args.excel_dir,
        doc_dir=args.doc_dir,
        bge_model_path=bge_model_path,
        save_path=args.save_path,
        budget=args.budget,
        per_table_quota=args.per_table_quota,
        device=args.device,
        require_cuda=args.require_cuda,
        document_chunk_size=args.document_chunk_size,
        document_chunk_overlap=args.document_chunk_overlap,
        row_budget=args.row_budget or None,
        row_per_table_quota=args.row_per_table_quota or None,
        max_row_chars=args.max_row_chars or None,
    )
    index.build()
    index.save()
    logger.info(f"Index saved to {args.save_path}.* in {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()
