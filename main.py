#!/usr/bin/env python3
"""
final_with_3_xai.py

Evolutionary kernel fuzzer with LLM guidance and triple XAI (SHAP, LIME, Permutation).

Usage:
  python3 final_with_3_xai.py --input snippets/data_utils.c --out results.csv --no-exec
  python3 final_with_3_xai.py --input snippets/data_utils.c \
      --syz-execprog /path/to/syz-execprog \
      --executor /path/to/syz-executor \
      --out results.csv

Dependencies:
  pip install pandas numpy scikit-learn shap lime requests
"""

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

try:
    import shap
    _SHAP_OK = True
except ImportError:
    _SHAP_OK = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    _LIME_OK = True
except ImportError:
    _LIME_OK = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL   = "http://192.168.56.1:11500"
OLLAMA_MODEL = "mistral:latest"

LLM_TIMEOUT       = 10
MAX_LLM_CALLS     = 150
DEFAULT_MAX_ITERS = 3
DEFAULT_MAX_QUEUE = 40

FEATURE_NAMES = [
    "Seed_Len", "Entropy", "Has_strcpy", "Has_memcpy", "Has_free",
    "Has_arith", "Line_Number", "Program_Syscalls",
    "Program_Has_write", "Program_Has_ioctl", "Risk_Keyword_Count",
]

RISK_KEYWORDS = [
    "overflow", "buffer", "use-after", "null", "memcpy",
    "strcpy", "uaf", "free", "panic", "oob",
]

_DETECTORS = [
    re.compile(r'\bstrcpy\s*\('),
    re.compile(r'\bmemcpy\s*\('),
    re.compile(r'\bfree\s*\('),
    re.compile(r'\bread\s*\(\s*[^\)]*NULL[^\)]*\)'),
    re.compile(r'\bopen\s*\('),
    re.compile(r'\b\w+\s*=\s*\w+\s*\+\s*\w+;'),
]

random.seed(42)
_llm_call_count = 0


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def _llm_budget_ok() -> bool:
    return _llm_call_count < MAX_LLM_CALLS


def call_ollama(prompt: str, timeout: int = LLM_TIMEOUT) -> Optional[str]:
    global _llm_call_count
    if not _llm_budget_ok():
        return None
    try:
        import requests
        _llm_call_count += 1
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            headers={"Content-Type": "application/json"},
            data=json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}),
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def parse_llm_json(raw: Optional[str]) -> Optional[Dict]:
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:]).strip()
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

def llm_get_spec(code: str, context: List[str]) -> Optional[Dict]:
    prompt = (
        f"Analyze this C kernel code. Return JSON with keys: "
        f"'spec' (bug description), 'seeds' (list of 2 strings), 'confidence' (0.0-1.0).\n"
        f"Code: {code}\nContext: {chr(10).join(context)}"
    )
    return parse_llm_json(call_ollama(prompt))


def llm_generate_prog(spec: str, code: str) -> Dict:
    prompt = (
        f"Generate a Syzkaller program (JSON) for this bug.\n"
        f"Bug: {spec}\nCode: {code}\n"
        f'Return JSON: {{"prog": [["syscall", [args...]], ...]}}'
    )
    result = parse_llm_json(call_ollama(prompt))
    if result and "prog" in result:
        return result
    return {"prog": [["write", [1, "0x41414141", 4]]]}


def llm_explain_crash(code: str, spec: str, log: str) -> Optional[Dict]:
    prompt = (
        f"Explain this kernel crash.\n"
        f"Code: {code}\nSpec: {spec}\nLog: {log}\n"
        f'Return JSON: {{"root_cause": "...", "fix": "...", "type": "..."}}'
    )
    return parse_llm_json(call_ollama(prompt))


def llm_mutate_seed(seed: str, log: str) -> List[str]:
    prompt = (
        f"Mutate this seed to trigger the crash more reliably.\n"
        f"Seed: {seed}\nLog: {log}\n"
        f'Return JSON: {{"mutated_seeds": ["new1", "new2"]}}'
    )
    result = parse_llm_json(call_ollama(prompt))
    if result and "mutated_seeds" in result:
        return result["mutated_seeds"]
    return []


# ---------------------------------------------------------------------------
# XAI
# ---------------------------------------------------------------------------

class MultiXAIExplainer:
    MIN_SAMPLES = 6

    def __init__(self) -> None:
        self.scaler: Optional[StandardScaler]       = StandardScaler() if _SKLEARN_OK else None
        self.model:  Optional[LogisticRegression]   = (
            LogisticRegression(solver="liblinear", warm_start=True) if _SKLEARN_OK else None
        )
        self.X:      List[List[float]] = []
        self.y:      List[int]         = []
        self.X_scaled: Optional[np.ndarray] = None
        self.lime_explainer: Optional[LimeTabularExplainer] = None

    def add_sample(self, features: Dict[str, float], crashed: bool) -> None:
        self.X.append([features.get(n, 0.0) for n in FEATURE_NAMES])
        self.y.append(int(crashed))

    def _fit(self) -> bool:
        if not _SKLEARN_OK or len(self.X) < self.MIN_SAMPLES:
            return False
        X_arr = np.array(self.X)
        try:
            self.X_scaled = self.scaler.fit_transform(X_arr)
            self.model.fit(self.X_scaled, np.array(self.y))
            return True
        except Exception:
            return False

    def _predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        X_s = self.scaler.transform(X_raw) if self.scaler else X_raw
        return self.model.predict_proba(X_s)

    def explain(self, features: Dict[str, float]) -> Dict[str, Any]:
        empty: Dict[str, Any] = {
            "shap":  {k: 0.0 for k in FEATURE_NAMES},
            "lime":  {"list": [], "score": 0.0},
            "perm":  {k: 0.0 for k in FEATURE_NAMES},
        }
        if not self._fit():
            return empty

        instance_raw    = np.array([features[n] for n in FEATURE_NAMES])
        instance_scaled = self.scaler.transform(instance_raw.reshape(1, -1))

        result = {
            "shap": self._run_shap(instance_scaled),
            "lime": self._run_lime(instance_raw),
            "perm": self._run_perm(instance_raw),
        }
        return result

    def _run_shap(self, instance_scaled: np.ndarray) -> Dict[str, float]:
        if not _SHAP_OK or self.X_scaled is None:
            return {k: 0.0 for k in FEATURE_NAMES}
        try:
            explainer = shap.LinearExplainer(
                self.model, self.X_scaled, feature_perturbation="interventional"
            )
            vals = np.array(explainer.shap_values(instance_scaled)).reshape(-1)
            return {FEATURE_NAMES[i]: float(vals[i]) for i in range(len(FEATURE_NAMES))}
        except Exception:
            return {k: 0.0 for k in FEATURE_NAMES}

    def _run_lime(self, instance_raw: np.ndarray) -> Dict[str, Any]:
        if not _LIME_OK:
            return {"list": [], "score": 0.0}
        try:
            if self.lime_explainer is None:
                self.lime_explainer = LimeTabularExplainer(
                    training_data=np.array(self.X),
                    feature_names=FEATURE_NAMES,
                    class_names=["NoCrash", "Crash"],
                    mode="classification",
                )
            exp   = self.lime_explainer.explain_instance(
                instance_raw, self._predict_proba, num_features=5
            )
            pairs = exp.as_list()
            return {"list": pairs, "score": sum(abs(v) for _, v in pairs)}
        except Exception:
            return {"list": [], "score": 0.0}

    def _run_perm(self, instance_raw: np.ndarray) -> Dict[str, float]:
        try:
            base_prob = self._predict_proba(instance_raw.reshape(1, -1))[0][1]
            col_means = np.mean(self.X, axis=0)
            scores: Dict[str, float] = {}
            for i, name in enumerate(FEATURE_NAMES):
                perturbed    = instance_raw.copy()
                perturbed[i] = col_means[i]
                new_prob     = self._predict_proba(perturbed.reshape(1, -1))[0][1]
                scores[name] = base_prob - new_prob
            return scores
        except Exception:
            return {k: 0.0 for k in FEATURE_NAMES}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _entropy(text: str) -> float:
    data = text.encode(errors="ignore")
    if not data:
        return 0.0
    probs = [data.count(b) / len(data) for b in set(data)]
    return float(-sum(p * np.log2(p) for p in probs))


def extract_features(
    code_line: str,
    prog: Dict,
    seed: str,
    line_number: int,
) -> Dict[str, float]:
    cl   = code_line.lower()
    ops  = [str(call[0]) for call in prog.get("prog", []) if call]
    text = cl + " " + json.dumps(prog)

    return {
        "Seed_Len":           float(len(seed)),
        "Entropy":            _entropy(seed),
        "Has_strcpy":         float("strcpy" in cl),
        "Has_memcpy":         float("memcpy" in cl),
        "Has_free":           float("free" in cl),
        "Has_arith":          float(bool(re.search(r"[\+\-\*\/%]", cl))),
        "Line_Number":        float(line_number),
        "Program_Syscalls":   float(len(ops)),
        "Program_Has_write":  float(any("write" in op.lower() for op in ops)),
        "Program_Has_ioctl":  float(any("ioctl" in op.lower() for op in ops)),
        "Risk_Keyword_Count": float(sum(1 for kw in RISK_KEYWORDS if kw in text.lower())),
    }


# ---------------------------------------------------------------------------
# Candidate scanning
# ---------------------------------------------------------------------------

def _subsystem_map(path: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    current = "default"
    for i, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
        if "subsystem:" in line.lower():
            parts = line.split(":", 1)
            if len(parts) > 1:
                current = parts[1].strip().split()[0]
        mapping[i] = current
    return mapping


def find_candidates(path: Path) -> List[Dict]:
    lines   = path.read_text(errors="ignore").splitlines()
    subs    = _subsystem_map(path)
    results = []
    for i, line in enumerate(lines, start=1):
        if any(pat.search(line) for pat in _DETECTORS):
            results.append({
                "line":      i,
                "code":      line.strip(),
                "context":   lines[max(0, i - 3) : min(len(lines), i + 2)],
                "subsystem": subs.get(i, "default"),
            })
    return results


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _seed_to_hex(seed: str) -> str:
    s = seed.strip()
    if s.startswith("0x"):
        try:
            bytes.fromhex(s[2:])
            return s
        except ValueError:
            pass
    return "0x" + s.encode(errors="ignore").hex()


def simulate_crash(spec: str, seed: str) -> bool:
    risk   = sum(1 for kw in RISK_KEYWORDS if kw in spec.lower())
    digest = hashlib.sha256(f"{spec}_{seed}".encode()).digest()[0] / 255.0
    return digest < min(0.02 + 0.1 * risk, 0.8)


def run_syzkaller(
    syz_execprog: Path,
    executor: Path,
    prog_file: Path,
) -> subprocess.CompletedProcess:
    cmd = [str(syz_execprog), "-executor", str(executor), str(prog_file)]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", "Timeout")
    except Exception as exc:
        return subprocess.CompletedProcess(cmd, 1, "", str(exc))


def inject_seed(prog: Dict, hex_seed: str) -> Dict:
    prog = json.loads(json.dumps(prog))
    for call in prog.get("prog", []):
        for i, arg in enumerate(call[1]):
            if isinstance(arg, str) and arg.startswith("0x"):
                call[1][i] = hex_seed
                return prog
    return prog


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_queue(candidates: List[Dict], use_llm: bool) -> List[Dict]:
    queue: List[Dict] = []
    default_prog = {"prog": [["write", [1, "0x41414141", 4]]]}

    for cand in candidates:
        spec_data = llm_get_spec(cand["code"], cand["context"]) if use_llm else None
        spec_data = spec_data or {"spec": "Heuristic check", "seeds": ["AAAA", "0x00"], "confidence": 0.5}

        prog = llm_generate_prog(spec_data["spec"], cand["code"]) if use_llm else default_prog

        for seed in spec_data.get("seeds", []):
            queue.append({
                "cand":  cand,
                "spec":  spec_data["spec"],
                "seed":  seed,
                "prog":  prog,
                "iters": 0,
            })
    return queue


def run_pipeline(cli_args: argparse.Namespace) -> None:
    input_path = Path(cli_args.input)
    if not input_path.exists():
        print(f"[ERR] File not found: {input_path}")
        return

    syz_mode = (
        not cli_args.no_exec
        and cli_args.syz_execprog
        and cli_args.executor
        and Path(cli_args.syz_execprog).exists()
        and Path(cli_args.executor).exists()
    )
    print(f"[INFO] {'Real execution' if syz_mode else 'Simulation'} mode.")

    candidates = find_candidates(input_path)
    print(f"[INFO] {len(candidates)} candidates found.")

    queue   = build_queue(candidates, cli_args.use_llm)[: cli_args.max_queue]
    xai     = MultiXAIExplainer()
    tmp_dir = Path(".tmp_syz")
    tmp_dir.mkdir(exist_ok=True)
    rows: List[Dict] = []

    print(f"[INFO] Queue size: {len(queue)}.")

    while queue:
        item  = queue.pop(0)
        cand  = item["cand"]
        seed  = item["seed"]
        iters = item["iters"]

        prog_with_seed = inject_seed(item["prog"], _seed_to_hex(seed))

        if syz_mode:
            prog_file = tmp_dir / f"prog_{os.getpid()}.json"
            prog_file.write_text(json.dumps(prog_with_seed))
            proc    = run_syzkaller(Path(cli_args.syz_execprog), Path(cli_args.executor), prog_file)
            log_out = proc.stdout + proc.stderr
            crashed = proc.returncode != 0 or "crash" in log_out.lower()
        else:
            crashed = simulate_crash(item["spec"], seed)
            log_out = "Simulated crash" if crashed else "Normal"

        features = extract_features(cand["code"], prog_with_seed, seed, cand["line"])
        xai.add_sample(features, crashed)

        xai_result:  Dict[str, Any] = {"shap": {}, "lime": {"list": [], "score": 0.0}, "perm": {}}
        explanation: Dict[str, str] = {}

        if crashed:
            print(f"  [!] CRASH  line={cand['line']}  seed='{seed}'")
            if cli_args.use_llm:
                explanation = llm_explain_crash(cand["code"], item["spec"], log_out) or {}
            xai_result = xai.explain(features)

            if iters < cli_args.max_iters:
                new_seeds = (llm_mutate_seed(seed, log_out) if cli_args.use_llm else [])
                new_seeds = new_seeds or [seed + "_mut1", seed + "X"]
                for ns in new_seeds[:2]:
                    queue.append({**item, "seed": ns, "prog": prog_with_seed, "iters": iters + 1})

        lime_list  = xai_result["lime"]["list"]
        shap_vals  = xai_result["shap"]
        perm_vals  = xai_result["perm"]

        row: Dict[str, Any] = {
            "Line":          cand["line"],
            "Subsystem":     cand["subsystem"],
            "Spec":          item["spec"],
            "Seed":          seed,
            "Crashed":       crashed,
            "Iterations":    iters,
            "Log_Snippet":   log_out[:100].replace("\n", " "),
            "LLM_Root_Cause": explanation.get("root_cause", ""),
            "LLM_Fix":        explanation.get("fix", ""),
            "LIME_Top1":     lime_list[0][0] if len(lime_list) > 0 else "",
            "LIME_Top2":     lime_list[1][0] if len(lime_list) > 1 else "",
            "LIME_Score":    xai_result["lime"]["score"],
            "SHAP_Top1":     max(shap_vals, key=lambda k: abs(shap_vals[k]), default=""),
            "PERM_Top1":     max(perm_vals, key=lambda k: abs(perm_vals[k]), default=""),
        }

        for name, val in features.items():
            row[f"Feat_{name}"] = val
        for name in FEATURE_NAMES:
            row[f"SHAP_{name}"] = shap_vals.get(name, 0.0)
            row[f"PERM_{name}"] = perm_vals.get(name, 0.0)

        rows.append(row)
        if len(queue) > cli_args.max_queue:
            queue = queue[: cli_args.max_queue]

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(cli_args.out, index=False)
        print(f"[DONE] {len(df)} rows → {cli_args.out}")
        print(df[["Line", "Seed", "Crashed", "LIME_Top1", "SHAP_Top1"]].head())
    else:
        print("[DONE] No results.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evolutionary kernel fuzzer with SHAP + LIME + Permutation XAI"
    )
    p.add_argument("--input",        "-i", required=True,           help="Path to C snippet file")
    p.add_argument("--out",          "-o", default="results_3xai.csv", help="Output CSV path")
    p.add_argument("--syz-execprog",                                 help="Path to syz-execprog binary")
    p.add_argument("--executor",                                     help="Path to syz-executor binary")
    p.add_argument("--no-exec",      action="store_true",            help="Force simulation mode")
    p.add_argument("--no-llm",       dest="use_llm", action="store_false", help="Disable LLM calls")
    p.add_argument("--max-iters",    type=int, default=DEFAULT_MAX_ITERS, help="Mutation depth")
    p.add_argument("--max-queue",    type=int, default=DEFAULT_MAX_QUEUE, help="Queue size cap")
    p.set_defaults(use_llm=True)
    return p.parse_args()


if __name__ == "__main__":
    if not _SKLEARN_OK:
        print("[WARN] scikit-learn not found — XAI disabled.")
    if not _LIME_OK:
        print("[WARN] LIME not found — LIME columns will be empty.")
    if not _SHAP_OK:
        print("[WARN] SHAP not found — SHAP columns will be empty.")

    run_pipeline(_parse_args())
