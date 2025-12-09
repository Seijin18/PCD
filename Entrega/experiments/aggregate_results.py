#!/usr/bin/env python3
import re
import csv
from pathlib import Path

RESULTS = Path("results")
OUT_CSV = RESULTS / "aggregated_large_results.csv"
OUT_STATS = RESULTS / "aggregated_large_stats.csv"


def parse_file(path: Path):
    text = path.read_text(errors="ignore")
    # mode from filename
    name = path.name
    mode = "unknown"
    threads = ""
    if "serial" in name:
        mode = "serial"
    elif "omp" in name:
        mode = "openmp"
        m = re.search(r"omp_(\d+)_", name)
        if m:
            threads = m.group(1)
        else:
            # try to find Threads: X in text
            m2 = re.search(r"Threads[:\s]+(\d+)", text)
            if m2:
                threads = m2.group(1)
    elif "cuda" in name:
        mode = "cuda"

    # iterations: prefer explicit "executadas: N" else count iteration lines
    iters = None
    m = re.search(r"executad[ao]s[:\s]+(\d+)", text, re.IGNORECASE)
    if m:
        iters = int(m.group(1))
    else:
        # count lines like "Itera... <num>: SSE"
        iters = len(re.findall(r"Itera[^\n]*\d+[:\)]", text, re.IGNORECASE))

    # SSE final
    sse = ""
    m = re.search(r"SSE final[:\s]+([0-9Ee.+-]+)", text)
    if m:
        sse = float(m.group(1))

    # Time total patterns
    time_ms = ""
    m = re.search(r"Tempo total[:\s]+([0-9.]+)\s*ms", text)
    if m:
        time_ms = float(m.group(1))
    else:
        # CUDA TOTAL line
        m = re.search(r"TOTAL[:\s]+([0-9.]+)\s*ms", text)
        if m:
            time_ms = float(m.group(1))

    # CUDA detailed
    h2d = kernel = d2h = throughput = time_per_iter = ""
    m = re.search(r"Transfer H2D[:\s]+([0-9.]+)\s*ms", text)
    if m:
        h2d = float(m.group(1))
    m = re.search(r"Kernels[:\s]+([0-9.]+)\s*ms", text)
    if m:
        kernel = float(m.group(1))
    m = re.search(r"Transfer D2H[:\s]+([0-9.]+)\s*ms", text)
    if m:
        d2h = float(m.group(1))
    m = re.search(r"Throughput[:\s]+([0-9.]+)\s*M pontos/segundo", text)
    if m:
        throughput = float(m.group(1))
    m = re.search(r"Tempo/itera[^\n:]*[:\s]+([0-9.]+)\s*ms", text)
    if m:
        time_per_iter = float(m.group(1))

    return {
        "file": name,
        "mode": mode,
        "threads": threads,
        "iterations": iters,
        "sse": sse,
        "time_ms": time_ms,
        "h2d_ms": h2d,
        "kernel_ms": kernel,
        "d2h_ms": d2h,
        "throughput_mpts_sec": throughput,
        "time_per_iter_ms": time_per_iter,
    }


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    files = sorted(RESULTS.glob("*_large_run*.txt"))
    if not files:
        print("No large-run logs found in results/. Run experiments first.")
        return 1

    rows = []
    for f in files:
        print("Parsing", f)
        rows.append(parse_file(f))

    # write per-run csv
    with OUT_CSV.open("w", newline="") as csvf:
        writer = csv.DictWriter(
            csvf,
            fieldnames=[
                "file",
                "mode",
                "threads",
                "iterations",
                "sse",
                "time_ms",
                "h2d_ms",
                "kernel_ms",
                "d2h_ms",
                "throughput_mpts_sec",
                "time_per_iter_ms",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # compute grouped statistics
    # group by mode + threads
    stats = {}
    for r in rows:
        key = (r["mode"], str(r["threads"]))
        stats.setdefault(key, {"times": [], "sses": [], "throughputs": []})
        if r["time_ms"] != "":
            stats[key]["times"].append(float(r["time_ms"]))
        if r["sse"] != "":
            stats[key]["sses"].append(float(r["sse"]))
        if r["throughput_mpts_sec"] != "":
            stats[key]["throughputs"].append(float(r["throughput_mpts_sec"]))

    with OUT_STATS.open("w", newline="") as sf:
        writer = csv.writer(sf)
        writer.writerow(
            [
                "mode",
                "threads",
                "n_runs",
                "time_mean_ms",
                "time_std_ms",
                "sse_mean",
                "sse_std",
                "throughput_mean_mpts",
                "throughput_std",
            ]
        )
        for (mode, threads), vals in sorted(stats.items()):
            import statistics as st

            n = max(len(vals["times"]), len(vals["sses"]), len(vals["throughputs"]))
            time_mean = st.mean(vals["times"]) if vals["times"] else ""
            time_std = st.pstdev(vals["times"]) if len(vals["times"]) > 0 else ""
            sse_mean = st.mean(vals["sses"]) if vals["sses"] else ""
            sse_std = st.pstdev(vals["sses"]) if len(vals["sses"]) > 0 else ""
            thr_mean = st.mean(vals["throughputs"]) if vals["throughputs"] else ""
            thr_std = (
                st.pstdev(vals["throughputs"]) if len(vals["throughputs"]) > 0 else ""
            )
            writer.writerow(
                [
                    mode,
                    threads,
                    n,
                    time_mean,
                    time_std,
                    sse_mean,
                    sse_std,
                    thr_mean,
                    thr_std,
                ]
            )

    print("Aggregated CSV written to", OUT_CSV)
    print("Aggregated stats written to", OUT_STATS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
