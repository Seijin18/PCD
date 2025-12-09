#!/usr/bin/env python3
"""Parse a single run TXT log and append a machine-readable CSV row.

This script is tolerant to small formatting differences and provides
fallbacks (e.g. use MPI total time as time_ms when time_ms isn't present).
"""
import argparse
import csv
import re
from pathlib import Path


def find_last_float(patterns, text):
    for p in patterns:
        m = re.findall(p, text, flags=re.IGNORECASE)
        if m:
            return m[-1]
    return ""


def parse_log(text):
    out = {}
    # Final SSE: last occurrence of SSE = <float>
    sse_matches = re.findall(r"SSE\s*=\s*([0-9eE+\-.]+)", text)
    out["sse_final"] = sse_matches[-1] if sse_matches else ""

    # Iterations: look for summary first, else fallback to count of SSE lines
    it_patterns = [
        r"Iter[a-zA-Z\W]{0,30}executad[^:\n]*:\s*(\d+)",
        r"Itera[^:\n]*:\s*(\d+)\b",
        r"Itera[^:\n]*s executadas:\s*(\d+)",
    ]
    it = None
    for p in it_patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            it = m.group(1)
            break
    if not it:
        # fallback: count SSE iteration lines
        it = str(len(sse_matches)) if sse_matches else ""
    out["iterations"] = it

    # Timing: try several patterns (Tempo total, TOTAL, Tempo: ... ms)
    time_patterns = [
        r"Tempo total:\s*([0-9.]+)\s*ms",
        r"TOTAL:\s*([0-9.]+)\s*ms",
        r"Tempo:\s*([0-9.]+)\s*ms",
        r"Tempo/itera[^:]*:\s*([0-9.]+)\s*ms",
    ]
    out["time_ms"] = find_last_float(time_patterns, text)

    # CUDA detailed timings
    out["h2d_ms"] = find_last_float(
        [r"Transfer H2D:\s*([0-9.]+)\s*ms", r"H2D:\s*([0-9.]+)\s*ms"], text
    )
    out["kernel_ms"] = find_last_float(
        [r"Kernels:\s*([0-9.]+)\s*ms", r"Kernel:\s*([0-9.]+)\s*ms"], text
    )
    out["d2h_ms"] = find_last_float(
        [r"Transfer D2H:\s*([0-9.]+)\s*ms", r"D2H:\s*([0-9.]+)\s*ms"], text
    )

    # Throughput (assumes 'M pontos/segundo' or 'M pontos' etc.)
    thr = find_last_float(
        [r"Throughput:\s*([0-9.]+)\s*M", r"Throughput:\s*([0-9.]+)"], text
    )
    out["throughput_millions_per_s"] = thr

    # MPI timings
    out["mpi_total_ms"] = find_last_float(
        [r"total_algorithm_time=([0-9.]+)\s*ms"], text
    )
    out["mpi_comm_total_ms"] = find_last_float(
        [r"comm_total[^=]*=([0-9.]+)\s*ms"], text
    )
    out["mpi_comm_avg_ms"] = find_last_float(
        [r"comm_avg_per_rank=([0-9.]+)\s*ms"], text
    )

    # Block size (common in CUDA output)
    bs = re.search(r"block_size\s*=\s*(\d+)", text)
    if not bs:
        # also accept 'Block size: 512' style
        bs = re.search(r"Block size[:\s]+(\d+)", text, flags=re.IGNORECASE)
    out["block_size"] = bs.group(1) if bs else ""

    return out


def append_csv(csv_path: Path, row: dict, fieldnames):
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", "-l", required=True, help="Path to TXT log file to parse")
    ap.add_argument(
        "--csv", "-c", default="results/run_summary.csv", help="CSV file to append"
    )
    ap.add_argument("--mode", help="mode (serial|omp|cuda|mpi)")
    ap.add_argument("--threads", help="number of threads (for OpenMP)")
    ap.add_argument("--label", help="optional label")
    args = ap.parse_args()

    log_path = Path(args.log)
    csv_path = Path(args.csv)
    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        return 1

    text = log_path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_log(text)

    # Prepare output row
    mode = args.mode if args.mode else ""
    if not mode:
        name = log_path.name.lower()
        if "cuda" in name:
            mode = "cuda"
        elif "mpi" in name:
            mode = "mpi"
        elif "omp" in name:
            mode = "omp"
        elif "serial" in name:
            mode = "serial"

    row = {
        "file": str(log_path.name),
        "mode": mode,
        "threads": args.threads or "",
        "label": args.label or "",
        "iterations": parsed.get("iterations", ""),
        "sse_final": parsed.get("sse_final", ""),
        "time_ms": parsed.get("time_ms", ""),
        "h2d_ms": parsed.get("h2d_ms", ""),
        "kernel_ms": parsed.get("kernel_ms", ""),
        "d2h_ms": parsed.get("d2h_ms", ""),
        "throughput_millions_per_s": parsed.get("throughput_millions_per_s", ""),
        "mpi_total_ms": parsed.get("mpi_total_ms", ""),
        "mpi_comm_total_ms": parsed.get("mpi_comm_total_ms", ""),
        "mpi_comm_avg_ms": parsed.get("mpi_comm_avg_ms", ""),
        "block_size": parsed.get("block_size", ""),
    }

    # If time_ms is empty but mpi_total_ms is present, use that for plotting consistency
    if (not row["time_ms"] or str(row["time_ms"]).strip() == "") and parsed.get(
        "mpi_total_ms"
    ):
        row["time_ms"] = parsed.get("mpi_total_ms")

    fieldnames = [
        "file",
        "mode",
        "threads",
        "label",
        "iterations",
        "sse_final",
        "time_ms",
        "h2d_ms",
        "kernel_ms",
        "d2h_ms",
        "throughput_millions_per_s",
        "mpi_total_ms",
        "mpi_comm_total_ms",
        "mpi_comm_avg_ms",
        "block_size",
    ]
    append_csv(csv_path, row, fieldnames)
    print(f"Appended metrics from {log_path} to {csv_path}")


if __name__ == "__main__":
    raise SystemExit(main())
