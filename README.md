# Every Answer Has Its Own Path: Agentic Routing for Retrieval-Augmented Table Question Answering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Repository for the EMNLP 2026 paper _"Every Answer Has Its Own Path: Agentic Routing for Retrieval-Augmented Table Question Answering"_.

![ATR architecture](./figures/architecture.png)

## Introduction

- We identify two failure modes of existing agentic table-text QA systems: they commit to a single execution strategy at the question level, and they suffer a **soft-retrieval / hard-execution gap**, where embedding retrieval ranks the correct cell as a candidate yet SQL execution still misses the row because the question's surface form never matches the stored value.
- We propose **ATR (AgenticTableRAG)**, which decomposes a question into sub-queries and routes each one to `TEXT` / `RETRIEVE` / `SQL` / `HYBRID` over a shared 5-view index. A **HybridValueLinker** grounds entity mentions to values that exist in the cell index before any SQL is issued, and a verifier rejects weak sub-answers and re-invokes the router with the failed-route history.
- The routing policy is distilled into a **DistilBERT student** that recovers 98.6% of the LLM teacher's decisions at zero per-sub-query LLM cost. A single configuration leads on token F1 across HybridQA, TAT-QA, and WTQ, and transfers unchanged to MultiHiertt and SPARTA, which the router never saw.

## Method overview

ATR builds shared offline indices and answers questions through an iterative
loop. The entry point is `AgenticTableRAGAgent.run_single` in
[`online/main.py`](online/main.py).

1. **Decompose:** generate the next structured sub-query from the question and
   previous sub-answers. Its metadata includes required modalities, expected
   operator, entity mentions, a global-table-view flag, and residual uncertainty.
2. **Retrieve:** retrieve document/table chunks and restore the associated table
   schema through the chunk-to-table mapping.
3. **Route and execute:** choose one of the four primitives below.
4. **Verify:** judge the answer against its evidence. A rejected answer updates
   the question-wide failed-route history and triggers re-routing, with up to
   three further attempts for that sub-query.
5. **Stop or continue:** stop on `TERMINATE`, on a supported answer with residual
   uncertainty below the threshold, or when the iteration budget is exhausted.
   Optional final synthesis combines at least two verifier-accepted sub-answers.

### Shared views and execution primitives

| View | Indexed content | Implementation |
| --- | --- | --- |
| 1 | Passage chunks | `DocumentRetriever` |
| 2 | Table chunks and their source/schema mapping | `DocumentRetriever` |
| 3 | Column names, types, and examples | `SchemaIndex` |
| 4 | Cell values and row context | `CellIndex`, `RowIndex` |
| 5 | Relational tables | External SQL service |

| Primitive | Execution |
| --- | --- |
| `TEXT` | Answer from retrieved passage and table chunks. |
| `RETRIEVE` | Ground entity mentions to cells, retrieve row context, and synthesize an answer without SQL. |
| `SQL` | Retrieve schema and execute NL2SQL without entity-value bindings. |
| `HYBRID` | Use HybridValueLinker to ground entity mentions and execute SQL constrained by retrieved columns and values. |

**HybridValueLinker** retrieves candidate cells and asks the LLM to select a
canonical value and column. Failed grounding uses an unconstrained / fuzzy
`LIKE` / `TEXT` fallback ladder, conditioned on prior failures. Candidate table
and column provenance is checked against the retrieved schema. The SQL executor
validates column scope and required value predicates before accepting a result;
the service adapter enforces the same constraints before database execution.

**Failure history** distinguishes the router's `requested_route` from the
`effective_route` that actually answered. For example, a HYBRID request can
fall back to TEXT inside ValueLinker. Subsequent routing records the effective
failed primitive. The learned router makes no LLM calls; the other prompted
components, SQL repairs, and final synthesis still incur LLM calls.

## Repository layout

```text
ATR/
├── online/
│   ├── main.py                # Algorithm 1 and batch inference
│   ├── decomposer.py          # Next SubQuery and TERMINATE
│   ├── parsing.py             # Shared structured-response parsing
│   ├── router.py              # Heuristic / LLM / learned / fixed routing
│   ├── value_linker.py        # HybridValueLinker
│   ├── constrained_sql.py     # (C, V*)-constrained SQL execution
│   └── verifier.py            # Evidence fusion and answer verification
├── offline/
│   ├── multiview_index.py     # Text, table, schema, cell, and row indices
│   └── reranking.py           # Cross-encoder candidate reranking
├── clients/                   # LLM, SQL, embedding, and table utilities
├── tools/
│   └── train_router.py        # Teacher labels and student training
├── figures/
│   └── architecture.png
├── build_index.py             # Offline index CLI
├── config.py                  # Backbones and default hyperparameters
├── prompt.py                  # All ATR prompts
├── sql_service_guard.py       # Pre-execution SQL constraint adapter
├── requirements.txt
├── README.md
└── LICENSE
```

## Setup

### Environment

Python 3.10 or later:

```bash
git clone https://github.com/ayoung206/ATR.git
cd ATR
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Models and credentials

1. **LLM:** the default is Gemini 2.5 Flash through Vertex AI. Place your
   service-account JSON at `./vertexai.json`, or set
   `VERTEXAI_CREDENTIALS_PATH` to its absolute path. Backbone keys and alternative
   endpoint settings are defined in [`config.py`](config.py).
2. **Retrieval:** place [BGE-M3](https://huggingface.co/BAAI/bge-m3) and
   [BGE-reranker-v2-M3](https://huggingface.co/BAAI/bge-reranker-v2-m3) in the
   directories below. `--bge_dir` points to their parent directory.
3. **Learned router:** download `atr-router.zip` from the
   [ATR router release](https://github.com/ayoung206/ATR/releases/tag/atr-router)
   and extract its checkpoint files into `models/atr_router/`. Alternatively,
   train a student using the commands under **Router training**.

```text
models/
├── bge-m3/
├── bge-reranker-v2-m3/
└── atr_router/
    ├── config.json
    ├── model.safetensors
    └── ...                    # Tokenizer files from the checkpoint
```

The released router uses input format **v1**: question/metadata features enter
the classifier, and failed routes are masked at its output. The current training
command produces **v2**, which also includes detailed schema and failure history
in the classifier input. The loader preserves the format recorded by each
checkpoint; a newly trained v2 model is a different model from the released v1
checkpoint.

### Datasets and input format

Obtain the datasets from their original releases and prepare the tables,
passages, and question files locally. This repository contains the core ATR
implementation; dataset preparation, experiment artifacts, and benchmark
scoring are maintained separately.

| Benchmark | Source | Input characteristics |
| --- | --- | --- |
| HybridQA | [Official repository](https://github.com/wenhuchen/HybridQA) | Wikipedia tables and linked passages |
| TAT-QA | [Official repository](https://github.com/NExTplusplus/TAT-QA) | Financial tables and explanatory paragraphs |
| WikiTableQuestions | [Official project](https://ppasupat.github.io/WikiTableQuestions/) | Standalone tables; no passage input |

Retain the upstream datasets' attribution and follow their source-content
terms. ATR's MIT license covers this implementation, not downloaded datasets.

Example prepared layout:

```text
data/
├── tables/                    # .xlsx or .csv files
├── passages/                  # .json or .txt files; empty for table-only QA
└── questions.json             # JSON array, or use a .jsonl question file
```

A question record has this form:

```json
{
  "question_id": "q1",
  "question": "Which team won the most games?",
  "table_id": "teams"
}
```

`question` is the question text, `question_id` identifies the example, and
`table_id` is a known source-table hint corresponding to the prepared table.
The hint is passed to the decomposer and used during document selection; it
is not merely an output label. Row retrieval is scoped to the first selected
table chunk, while SQL scope is derived from retrieved schema. Keep the table
hint condition consistent when comparing runs.

Gold answers are not needed for normal inference. An optional `answer-text`
field is carried into the output and is used by the oracle-verifier diagnostic
when explicitly enabled. Training and validation question files must be
separated at the source-question level before generating sub-query labels.

### SQL service

`SQL` and `HYBRID` require an external NL2SQL/database service, such as
[TableRAG](https://github.com/yxh-y/TableRAG). Tables must be ingested into
that service separately. Configure its LLM to match the inference backbone.

```bash
export SQL_SERVICE_URL=http://127.0.0.1:5000/get_tablerag_response
export ATR_SQL_BACKBONE=gemini
```

The client sends constrained requests to the base URL plus `/atr-guarded-v1`.
The external Flask service can register this endpoint with
`sql_service_guard.register_guarded_route(app, process_request)`.
Its `process_request(table_name_list, query, sql_validator=...)` implementation
must call `sql_validator(sql)` on the exact generated SQL before database
execution, abort on validation failure, and return a dictionary containing
`sql_str` and complete `sql_execution_result` rows. The adapter acknowledges
validation with `atr_guard_version=1`. An unmodified TableRAG endpoint does
not implement this contract. `call_sql_llm` in the same module connects the
service's prompts to the configured ATR backbone.

## Usage

All commands below run from the repository root. Paths refer to locally
prepared inputs and model checkpoints.

### 1. Build the offline indices

```bash
python -m build_index \
    --excel_dir data/tables \
    --doc_dir data/passages \
    --bge_dir models \
    --save_path index/multiview \
    --document_chunk_size 512 \
    --document_chunk_overlap 64 \
    --budget 10000
```

For table-only datasets such as WTQ, create an empty `data/passages/` directory
and pass it as `--doc_dir`. The index builder constructs Views 1--4; ingest
relational tables into the SQL service separately for View 5.

Index construction uses BGE-M3. Online retrieval recalls six times the requested
result count, then reranks candidates with BGE-reranker-v2-M3. The cell budget
is a dataset-wide cap of 10,000 entries, with at most 50 entries per table by
default; it does not guarantee cell coverage of every table. Row indexing
includes every source row by default. Source file order can affect a capped
index, so reuse the same prepared inputs and index when comparing runs.

The loader checks the saved format, build settings, metadata counts, and FAISS
components. Rebuild legacy or partially rebuilt indices with this index builder.

### 2. Run online inference

```bash
python -m online.main \
    --backbone gemini \
    --data_file_path data/questions.json \
    --index_path index/multiview \
    --bge_dir models \
    --router_type learned \
    --router_model_path models/atr_router \
    --max_iter 5 --max_workers 2 \
    --device cuda --router_device cuda --require_cuda \
    --final_synthesis \
    --save_file_path output/answers.jsonl
```

For CPU execution, replace the device arguments with
`--device cpu --router_device cpu` and omit `--require_cuda`.
Without a SQL service, run a fixed text route:

```bash
python -m online.main \
    --backbone gemini \
    --data_file_path data/questions.json \
    --index_path index/multiview \
    --bge_dir models \
    --router_type fixed --force_route TEXT --no_escalation \
    --device cpu \
    --save_file_path output/text_answers.jsonl
```

### 3. Read outputs and traces

Inference appends a timestamp to the requested output filename. For example,
`output/answers.jsonl` becomes `output/answers_YYYYMMDD_HHMMSS.jsonl`.
Use the actual path reported by the run. `--rerun` uses the exact supplied path
and resumes records by question text.

Each output record preserves the input fields and adds
`agentic_tablerag_answer`. Add `--emit_trace` to include `atr_trace`, with
per-attempt router inputs, requested/effective routes, verifier verdicts,
produced answers, and failure histories. This lets you follow one question
through decomposition, execution, and re-routing.

The loop retains failed sub-answers for subsequent decomposition, but final
synthesis only receives accepted sub-answers. If synthesis is disabled, the
agent returns its last verified valid answer; if none exists, it returns
`not found`.

### 4. Configuration and ablations

| Setting | Default / behavior |
| --- | --- |
| `--max_iter` | 5 outer decomposition iterations |
| `--verifier_threshold` | 0.1 residual-uncertainty threshold |
| `--max_workers` | 2 concurrent questions; each question's sub-queries remain sequential |
| `--router_type` | CLI default is `heuristic`; pass `learned` for the released student |
| `--final_synthesis` | Off unless supplied; combine at least two accepted sub-answers |
| Schema / cell / row top-k | 5 / 15 / 10; environment settings in `config.py` |
| `--rerank_candidate_multiplier` | 6 dense candidates per requested result |
| `--use_schema_preview` | Opt-in schema preview for the decomposer |
| `--use_markdown_fusion` | Opt-in conditional SQL scalar fusion |

Useful ablation flags:

- `--router_type llm`: use the prompted LLM router.
- `--router_type fixed --force_route HYBRID`: fix the initial routing choice.
  `--fixed_escalate_chain standard` enables retries through the remaining
  primitives; without that option, the fixed router keeps its chosen route.
  ValueLinker's internal TEXT fallback is separate from router escalation.
- `--no_escalation`: disable verifier-triggered route retries.
- `--no_decomposition`: use the raw question and cap the loop at one iteration.
- `--no_value_linker`: omit entity-value bindings while retaining column constraints.
- `--no_reranker`: disable cross-encoder reranking.
- `--decomposer_backbone` / `--verifier_backbone`: change only the corresponding
  prompted component.
- `--oracle_verifier`: after normal inference, select a generated candidate that
  matches the gold answer, if one exists. This is a post-hoc candidate-selection
  diagnostic, not an online replacement of each verifier verdict.

For a full backbone swap, change `--backbone` and configure/restart the SQL
service with the same `ATR_SQL_BACKBONE`. Changing the client flag alone does
not change the external service's model. See `config_mapping` for backbone keys.

### 5. Train the learned router

Prepare `data/router_train.json` and `data/router_valid.json` from disjoint
HybridQA training questions, with corresponding tables under `data/train_tables/`.
Keep every sub-query of a source question in the same split. Do not use downstream
dev questions or inference traces to train or select the router.

```bash
# Generate teacher labels from training questions.
python -m tools.train_router distill \
    --data_file data/router_train.json \
    --excel_dir data/train_tables \
    --backbone gemini \
    --out_file labels/router_train.jsonl

# Fine-tune the student.
python -m tools.train_router train \
    --oracle_file labels/router_train.jsonl \
    --output_dir models/router_trained \
    --epochs 5 --batch_size 32 --lr 2e-5

# Generate labels for separate held-out training questions.
python -m tools.train_router distill \
    --data_file data/router_valid.json \
    --excel_dir data/train_tables \
    --backbone gemini \
    --out_file labels/router_valid.jsonl

# Measure teacher agreement on the held-out labels.
python -m tools.train_router eval \
    --model_dir models/router_trained \
    --oracle_file labels/router_valid.jsonl
```

`distill` decomposes each question to its first sub-query and records the
teacher's initial choice plus up to three re-selections under simulated
cumulative failure histories. It does not execute or verify the candidate
routes. `train` uses class-weighted cross-entropy and an internal random
record-level validation split for checkpoint selection. That internal split
does not group records by source question. The separate question-level holdout
above is used for reporting agreement; `eval` scores every supplied record
without creating a split. Router agreement and downstream answer accuracy are
different measurements.

## Acknowledgements

ATR's `SQL` and `HYBRID` primitives execute against the Flask SQL service
released with **TableRAG** (Yu et al., EMNLP 2025),
https://github.com/yxh-y/TableRAG/. This repository ships no code from that
project as a standalone service; the supplied integration patch is applied to
its external checkout. Please cite their paper if you use it.

The multi-view index is built on **BGE-M3** (Chen et al., 2024) and the
benchmarks are HybridQA, TAT-QA, and WikiTableQuestions; cite those alongside
ATR when you report numbers.

## Citation

```bibtex
@inproceedings{kim2026atr,
  title     = {Every Answer Has Its Own Path: Agentic Routing for Retrieval-Augmented Table Question Answering},
  author    = {Kim, A Young and Shin, Jisu and Han, Donghee and Yi, Mun Yong},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026}
}
```
