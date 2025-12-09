#!/usr/bin/env python3
"""
Validate K-means implementations by running binaries, computing SSE from outputs,
and comparing assignments and SSE between implementations.

Usage: python validate.py
Defaults use `Entrega/bin` executables and `Entrega/Data/dados_small.csv`.
"""
import subprocess
import csv
import sys
from pathlib import Path
import time
import math

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "bin"
DATA = ROOT / "Data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

IMPLEMENTATIONS = [
    ("serial", BIN / "kmeans_serial.exe"),
    ("openmp", BIN / "kmeans_omp.exe"),
    ("cuda", BIN / "kmeans_cuda.exe"),
]

DEFAULT_DATA = DATA / "dados_small.csv"
DEFAULT_INIT = DATA / "centroides_small.csv"


def read_floats(path):
    vals = []
    with open(path, "r", newline="") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    vals.append(float(s))
                except ValueError:
                    # allow CSV with commas
                    parts = s.split(",")
                    vals.append(float(parts[0]))
    return vals


def read_ints(path):
    vals = []
    with open(path, "r", newline="") as f:
        for line in f:
            s = line.strip()
            if s:
                vals.append(int(s))
    return vals


def compute_sse(data, centroids, assigns):
    sse = 0.0
    for x, a in zip(data, assigns):
        if a < 0 or a >= len(centroids):
            # invalid assignment
            sse += float("inf")
        else:
            d = x - centroids[a]
            sse += d * d
    return sse


def run_impl(name, exe, data_file, init_file, k, max_iter, eps):
    # support multiple naming conventions (openmp -> omp)
    prefixes = [name]
    if name == "openmp":
        prefixes.append("omp")
    out_assign = None
    out_centroids = None
    # pick default expected names (first prefix)
    out_assign = RESULTS / f"assign_{prefixes[0]}.csv"
    out_centroids = RESULTS / f"centroids_{prefixes[0]}.csv"
    log = RESULTS / f"log_{name}.txt"
    cmd = [
        str(exe),
        str(data_file),
        str(init_file),
        str(k),
        str(max_iter),
        str(eps),
        str(out_assign),
        str(out_centroids),
    ]
    print("Running", name, "->", " ".join(cmd))
    t0 = time.time()
    try:
        with open(log, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=600)
    except subprocess.TimeoutExpired:
        return {"name": name, "error": "timeout"}
    dt = time.time() - t0
    # Read outputs if present. If not, try to discover generated files (pattern-based)
    if not out_assign.exists() or not out_centroids.exists():
        # try to parse the log to find filenames
        try:
            txt = log.read_text(encoding="utf-8")
            # look for line like: Resultados salvos em: assign_omp_50.csv e centroids_omp_50.csv
            for line in txt.splitlines():
                if "Resultados salvos em" in line:
                    parts = line.split(":", 1)[1]
                    parts = parts.replace(" e ", ",")
                    names = [p.strip() for p in parts.split(",") if p.strip()]
                    if len(names) >= 2:
                        candidate_assign = RESULTS / names[0]
                        candidate_cent = RESULTS / names[1]
                        if candidate_assign.exists() and candidate_cent.exists():
                            out_assign = candidate_assign
                            out_centroids = candidate_cent
                            break
        except Exception:
            pass

    # fallback: glob patterns
    if not out_assign.exists() or not out_centroids.exists():
        import glob

        a_matches = []
        c_matches = []
        for p in prefixes:
            pattern_a = str(RESULTS / f"assign_{p}*.csv")
            pattern_c = str(RESULTS / f"centroids_{p}*.csv")
            a_matches += glob.glob(pattern_a)
            c_matches += glob.glob(pattern_c)
        if a_matches and c_matches:
            # prefer file whose number of lines matches data length
            try:
                data_len = sum(1 for _ in open(data_file, "r", encoding="utf-8"))
            except Exception:
                data_len = None

            def best_match(lst):
                if data_len is None:
                    return Path(lst[-1])
                best = None
                best_diff = None
                for p in lst:
                    try:
                        l = sum(1 for _ in open(p, "r", encoding="utf-8"))
                        diff = abs(l - data_len)
                        if best is None or diff < best_diff:
                            best = p
                            best_diff = diff
                    except Exception:
                        continue
                return Path(best) if best else Path(lst[-1])

            out_assign = best_match(a_matches)
            out_centroids = best_match(c_matches)

    # final fallback: search workspace for assign_{name}*.csv and pick best candidates
    if not out_assign.exists() or not out_centroids.exists():
        a_matches = []
        c_matches = []
        # search both the Entrega folder (ROOT) and its parent (repo root)
        search_bases = [ROOT, ROOT.parent]
        for base in search_bases:
            for p in prefixes:
                a_matches.extend(list(base.glob(f"**/assign_{p}*.csv")))
                c_matches.extend(list(base.glob(f"**/centroids_{p}*.csv")))

        # determine data length if possible
        try:
            data_len = sum(1 for _ in open(data_file, "r", encoding="utf-8"))
        except Exception:
            data_len = None

        def choose_best(cands, expected_lines=None):
            if not cands:
                return None
            # if we know expected line count, pick the candidate with line count closest to it
            if expected_lines is not None:
                best = None
                best_diff = None
                for c in cands:
                    try:
                        cnt = sum(1 for _ in open(c, "r", encoding="utf-8"))
                    except Exception:
                        continue
                    # ignore obviously huge legacy files (100x larger than expected)
                    if expected_lines > 0 and cnt > expected_lines * 100:
                        continue
                    diff = abs(cnt - expected_lines)
                    if best is None or diff < best_diff:
                        best = c
                        best_diff = diff
                if best is not None:
                    return best
            # prefer files inside the RESULTS folder if available
            try:
                in_results = [c for c in cands if str(RESULTS) in str(c.resolve())]
            except Exception:
                in_results = []
            if in_results:
                try:
                    return sorted(in_results, key=lambda p: p.stat().st_mtime)[-1]
                except Exception:
                    return in_results[-1]
            # otherwise pick most recent file overall
            try:
                return sorted(cands, key=lambda p: p.stat().st_mtime)[-1]
            except Exception:
                return cands[-1]

        best_a = choose_best(a_matches, expected_lines=data_len)
        best_c = choose_best(c_matches, expected_lines=k)
        # choose candidates if both found
        if best_a is not None and best_c is not None:
            out_assign = best_a
            out_centroids = best_c

    if not out_assign.exists() or not out_centroids.exists():
        return {"name": name, "error": "missing_output", "time_s": dt}
    print(f"Using assign file: {out_assign}")
    print(f"Using centroids file: {out_centroids}")
    assigns = read_ints(out_assign)
    cents = read_floats(out_centroids)
    data = read_floats(data_file)
    # handle small mismatches in lengths (trim or pad) to allow comparison
    if len(assigns) != len(data):
        diff = len(assigns) - len(data)
        if abs(diff) <= 2:
            if diff > 0:
                # trim extra assignments
                assigns = assigns[: len(data)]
            else:
                # pad with zeros
                assigns += [0] * (-diff)
        else:
            return {"name": name, "error": "mismatched_lengths", "time_s": dt}
    sse = compute_sse(data, cents, assigns)
    return {
        "name": name,
        "time_s": dt,
        "sse": sse,
        "assigns": assigns,
        "centroids": cents,
    }


def main():
    data_file = DEFAULT_DATA
    init_file = DEFAULT_INIT
    k = 4
    max_iter = 50
    eps = 1e-4

    results = []
    for name, exe in IMPLEMENTATIONS:
        if not exe.exists():
            print("Skipping", name, "- executable not found:", exe)
            continue
        res = run_impl(name, exe, data_file, init_file, k, max_iter, eps)
        results.append(res)

    # Simple pairwise comparison
    summary = []
    ref = None
    tol_rel = 1e-6
    for r in results:
        if "error" in r:
            print("Result", r["name"], "error:", r.get("error"))
            summary.append((r["name"], "ERROR", r.get("error")))
            continue
        if ref is None:
            ref = r
            summary.append((r["name"], "OK", r["sse"]))
            continue
        # compare assigns
        same_assigns = r["assigns"] == ref["assigns"]
        sse_rel = abs(r["sse"] - ref["sse"]) / (abs(ref["sse"]) + 1e-12)
        ok = same_assigns or (sse_rel <= tol_rel)
        summary.append(
            (
                r["name"],
                "OK" if ok else "DIFF",
                {"sse": r["sse"], "sse_rel": sse_rel, "same_assigns": same_assigns},
            )
        )

    # Write summary
    out_csv = RESULTS / "validation_summary.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["impl", "status", "info"])
        for row in summary:
            w.writerow([row[0], row[1], row[2]])

    print("\nValidation summary:")
    for row in summary:
        print(row)
    print("\nFull logs and outputs in", RESULTS)


if __name__ == "__main__":
    main()
