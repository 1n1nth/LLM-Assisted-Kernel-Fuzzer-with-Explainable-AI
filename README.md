# Evolutionary Kernel Fuzzer with LLM + Triple XAI

An evolutionary fuzzer for kernel C code that combines **Syzkaller-style program execution** with **LLM-guided seed generation** (via Ollama) and three complementary explainability methods — **SHAP**, **LIME**, and **Permutation Importance** — to surface which code features drive crash probability.

---

## How It Works

```
C source file
     │
     ▼
Candidate scanner  ──►  risky lines (strcpy, memcpy, free, OOB arith …)
     │
     ▼
LLM (Ollama)  ──►  bug spec + seeds + Syzkaller program JSON
     │
     ▼
Execution engine  ──►  syz-execprog + syz-executor  (or simulation)
     │
     ▼
XAI explainer  ──►  SHAP / LIME / Permutation scores per feature
     │
     ▼
results.csv  ──►  one row per (candidate line × seed × iteration)
```

On each crash the fuzzer: explains it with the LLM, scores it with all three XAI methods, then mutates the seed and re-queues it (up to `--max-iters` deep).

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) running locally with the `mistral:latest` model pulled
- *(Optional)* `syz-execprog` and `syz-executor` from [google/syzkaller](https://github.com/google/syzkaller) for real kernel execution

### Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Ollama Setup

The fuzzer talks to Ollama at `http://192.168.56.1:11500` by default (editable at the top of `main.py`).

```bash
# Install Ollama — https://ollama.com/download
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull mistral:latest

# Start the server (if not already running)
ollama serve
```

Change `OLLAMA_URL` and `OLLAMA_MODEL` in `main.py` to point to your instance.

---

## Syzkaller Binaries (Optional)

`syz-execprog` and `syz-executor` are **not included** in this repo — they are compiled, platform-specific binaries. Build them from source:

```bash
git clone https://github.com/google/syzkaller
cd syzkaller
make
# outputs land in bin/ — use bin/syz-execprog and bin/syz-executor
```

Requires Go 1.21+. See [syzkaller docs](https://github.com/google/syzkaller/blob/master/docs/linux/setup.md) for kernel and VM setup.

If you don't have the binaries, use `--no-exec` to run in simulation mode (see below).

---

## Usage

### Simulation mode (no binaries needed)

```bash
python3 main.py \
  --input snippets/data_utils.c \
  --out results.csv \
  --no-exec
```

### Real execution mode

```bash
python3 main.py \
  --input snippets/data_utils.c \
  --syz-execprog /path/to/syz-execprog \
  --executor    /path/to/syz-executor \
  --out results.csv
```

### Disable LLM (heuristic-only, faster)

```bash
python3 main.py \
  --input snippets/data_utils.c \
  --no-exec \
  --no-llm
```

### All flags

| Flag | Default | Description |
|---|---|---|
| `--input` / `-i` | *(required)* | Path to C source file to analyse |
| `--out` / `-o` | `results_3xai.csv` | Output CSV path |
| `--syz-execprog` | — | Path to `syz-execprog` binary |
| `--executor` | — | Path to `syz-executor` binary |
| `--no-exec` | off | Force simulation mode (ignores binaries) |
| `--no-llm` | off | Disable all Ollama calls; use heuristics only |
| `--max-iters` | `3` | Crash mutation depth |
| `--max-queue` | `40` | Max items in the fuzzing queue at once |

---

## Output CSV Columns

| Column | Description |
|---|---|
| `Line` | Source line number of the candidate |
| `Subsystem` | Kernel subsystem label parsed from source |
| `Spec` | Bug description (LLM or heuristic) |
| `Seed` | Input seed used for this run |
| `Crashed` | `True` / `False` |
| `Iterations` | Mutation depth of this seed |
| `Log_Snippet` | First 100 chars of executor output |
| `LLM_Root_Cause` | LLM crash explanation |
| `LLM_Fix` | LLM suggested fix |
| `LIME_Top1/2` | Top LIME feature contributions |
| `SHAP_Top1` | Feature with highest SHAP magnitude |
| `PERM_Top1` | Feature with highest permutation importance |
| `Feat_*` | Raw feature values |
| `SHAP_*` | Per-feature SHAP values |
| `PERM_*` | Per-feature permutation scores |

---

## Features Used by XAI

| Feature | Description |
|---|---|
| `Seed_Len` | Length of the seed string |
| `Entropy` | Shannon entropy of the seed |
| `Has_strcpy` | `strcpy` present in candidate line |
| `Has_memcpy` | `memcpy` present in candidate line |
| `Has_free` | `free` present in candidate line |
| `Has_arith` | Arithmetic operator in candidate line |
| `Line_Number` | Line number in source file |
| `Program_Syscalls` | Number of syscalls in generated program |
| `Program_Has_write` | `write` syscall in program |
| `Program_Has_ioctl` | `ioctl` syscall in program |
| `Risk_Keyword_Count` | Count of risk keywords (overflow, uaf, oob …) |

XAI kicks in once at least 6 samples have been collected. Before that, scores default to zero.

---

## Project Structure

```
.
├── main.py                  # Fuzzer entry point
├── requirements.txt         # Python dependencies
├── snippets/
│   └── data_utils.c         # Example C input file
└── README.md
```

---

## Limitations & Notes

- **LLM budget**: capped at 150 Ollama calls per run (`MAX_LLM_CALLS` in `main.py`).
- **XAI minimum**: SHAP/LIME/Permutation require ≥ 6 samples; they are silently disabled before that.
- **Real crashes**: actual kernel crashes require a properly configured Syzkaller VM/target. Running `syz-execprog` against a live kernel without that setup is unsafe — use simulation mode for development.
- **Seed determinism**: `random.seed(42)` is set at startup; results are reproducible in `--no-exec --no-llm` mode.

---

## License

MIT
