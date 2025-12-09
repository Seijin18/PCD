#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("results")
# Prefer an aggregated CSV, but fall back to the run_summary produced by the
# per-run exporter (save_run_csv.py)
CSV = RESULTS / "aggregated_large_results.csv"
if not CSV.exists():
    CSV = RESULTS / "run_summary.csv"
GRAPHS = RESULTS / "graphs"


def main():
    GRAPHS.mkdir(parents=True, exist_ok=True)
    if not CSV.exists():
        print(
            "Aggregated CSV not found. Run experiments/save_run_csv.py or make test-large first."
        )
        return 1
    print("DEBUG: reading aggregated CSV:", CSV)
    df = pd.read_csv(CSV)
    print("DEBUG: CSV loaded, rows=", len(df))
    print("DEBUG: CSV head:\n", df.head().to_string())

    # Prefer stats CSV for serial baseline if present
    stats_csv = RESULTS / "aggregated_large_stats.csv"
    serial_time = None
    if stats_csv.exists():
        sstats = pd.read_csv(stats_csv)
        print("DEBUG: stats CSV loaded, rows=", len(sstats))
        print("DEBUG: stats head:\n", sstats.head().to_string())
        srow = sstats[sstats["mode"] == "serial"]
        if not srow.empty:
            serial_time = float(srow["time_mean_ms"].iloc[0])
            print("DEBUG: serial_time from stats=", serial_time)

    # Normalize time_ms (drop empty)
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
    # If some rows (e.g. MPI) do not populate time_ms but have mpi_total_ms, use it
    if "mpi_total_ms" in df.columns:
        df["mpi_total_ms"] = pd.to_numeric(df["mpi_total_ms"], errors="coerce")
        df["time_ms"] = df["time_ms"].fillna(df["mpi_total_ms"])
    if serial_time is None:
        ser = df[df["mode"] == "serial"]
        if not ser.empty:
            serial_time = ser["time_ms"].mean()

    # Plot time by mode/thread
    plt.figure(figsize=(8, 5))

    # prepare labels: prefer explicit 'label' column (e.g., bs256), else construct
    def label_row(r):
        mode = str(r.get("mode", "")).strip().lower()
        # treat missing/NaN labels properly
        explicit_val = r.get("label", "")
        explicit = ""
        if pd.notna(explicit_val):
            explicit = str(explicit_val).strip()
        # OpenMP: use threads
        if mode in ("openmp", "omp"):
            th = r.get("threads", "")
            try:
                thv = int(th)
                return f"OpenMP-{thv}t"
            except Exception:
                return "OpenMP"
        # CUDA: prefer explicit label (bs32), else include block_size if present
        if mode == "cuda":
            if explicit:
                # avoid doubling if explicit already includes 'cuda'
                if "cuda" in explicit.lower():
                    return explicit
                return f"CUDA-{explicit}"
            bs = r.get("block_size", "")
            if bs and str(bs).strip():
                return f"CUDA-bs{int(float(bs))}"
            return "CUDA"
        # MPI: prefer threads (number of processes) then explicit label
        if mode == "mpi":
            th = r.get("threads", "")
            try:
                thv = int(th)
                return f"MPI-{thv}p"
            except Exception:
                if explicit:
                    return f"MPI-{explicit}"
                return "MPI"
        # Serial or fallback
        if explicit:
            return explicit.capitalize()
        if mode:
            return mode.capitalize()
        return "Run"

    df["label"] = df.apply(label_row, axis=1)
    print("DEBUG: labels to plot=", list(df.apply(label_row, axis=1)))
    # Ensure index is unique for plotting (append counter for duplicates)
    labels = df["label"].astype(str).tolist()
    seen = {}
    unique_labels = []
    for l in labels:
        if l in seen:
            seen[l] += 1
            unique_labels.append(f"{l} ({seen[l]})")
        else:
            seen[l] = 1
            unique_labels.append(l)
    df["plot_label"] = unique_labels
    df_plot = df.set_index("plot_label")
    print("DEBUG: df_plot time_ms:\n", df_plot["time_ms"].to_string())
    df_plot["time_ms"].plot(kind="bar", title="Execution time (ms) - large dataset")
    plt.ylabel("Time (ms)")
    plt.tight_layout()
    plt.savefig(GRAPHS / "time_by_mode.png")
    print("Saved", GRAPHS / "time_by_mode.png")

    # Throughput plot (if available). Accept either column name used in different scripts
    thr_col = None
    for c in ("throughput_millions_per_s", "throughput_mpts_sec", "throughput"):
        if c in df.columns:
            thr_col = c
            break
    if thr_col is not None:
        df[thr_col] = pd.to_numeric(df[thr_col], errors="coerce")
        if df[thr_col].notna().any():
            plt.figure(figsize=(8, 5))
            # Use the same plot_label index so bars align with time plot
            df_tmp = df.set_index("plot_label")[thr_col].dropna()
            df_tmp.plot(kind="bar", title="Throughput (M points/s) - large dataset")
            plt.ylabel("M points / s")
            plt.tight_layout()
            plt.savefig(GRAPHS / "throughput_by_mode.png")
            print("Saved", GRAPHS / "throughput_by_mode.png")

    # Speedup plot (relative to serial)
    if serial_time is not None:
        df_speed = df[df["mode"].str.lower() != "serial"].copy()
        df_speed["speedup"] = serial_time / df_speed["time_ms"]
        print("DEBUG: serial_time used for speedup=", serial_time)
        print(
            "DEBUG: speedup values:\n",
            df_speed[["plot_label", "time_ms", "speedup"]].to_string(index=False),
        )
        plt.figure(figsize=(8, 5))
        # Plot as bar to show each configuration separately
        df_speed_plot = df_speed.set_index("plot_label")["speedup"].dropna()
        df_speed_plot.plot(kind="bar", title="Speedup vs Serial (large dataset)")
        plt.xlabel("Configuration")
        plt.ylabel("Speedup")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(GRAPHS / "speedup_vs_serial.png")
        print("Saved", GRAPHS / "speedup_vs_serial.png")
    else:
        print("Serial baseline not found; skipping speedup plot")

    # Time per iteration plot (if available)
    if "time_per_iter_ms" in df.columns:
        df["time_per_iter_ms"] = pd.to_numeric(df["time_per_iter_ms"], errors="coerce")
        if df["time_per_iter_ms"].notna().any():
            plt.figure(figsize=(8, 5))
            df_tmp = df.set_index("label")["time_per_iter_ms"].dropna()
            df_tmp.plot(kind="bar", title="Time per iteration (ms) - large dataset")
            plt.ylabel("ms / iteration")
            plt.tight_layout()
            plt.savefig(GRAPHS / "time_per_iter_by_mode.png")
            print("Saved", GRAPHS / "time_per_iter_by_mode.png")

    # Save a small table CSV for manual inspection (time, throughput, mpi)
    out_table = RESULTS / "aggregated_table_for_report.csv"
    df.to_csv(out_table, index=False)
    print("Saved table", out_table)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
