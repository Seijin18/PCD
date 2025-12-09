#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("results")
CSV = RESULTS / "aggregated_large_results.csv"
GRAPHS = RESULTS / "graphs"


def main():
    GRAPHS.mkdir(parents=True, exist_ok=True)
    if not CSV.exists():
        print("Aggregated CSV not found. Run aggregate_results.py first.")
        return 1
    df = pd.read_csv(CSV)

    # Prefer stats CSV for serial baseline if present
    stats_csv = RESULTS / "aggregated_large_stats.csv"
    serial_time = None
    if stats_csv.exists():
        sstats = pd.read_csv(stats_csv)
        srow = sstats[sstats["mode"] == "serial"]
        if not srow.empty:
            serial_time = float(srow["time_mean_ms"].iloc[0])

    # Normalize time_ms (drop empty)
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="coerce")
    if serial_time is None:
        ser = df[df["mode"] == "serial"]
        if not ser.empty:
            serial_time = ser["time_ms"].mean()

    # Plot time by mode/thread
    plt.figure(figsize=(8, 5))

    # prepare labels
    def label_row(r):
        if r["mode"] == "openmp" and pd.notna(r["threads"]) and r["threads"] != "":
            return f"OpenMP-{int(r['threads'])}t"
        else:
            return r["mode"].capitalize()

    df["label"] = df.apply(label_row, axis=1)
    df_plot = df.set_index("label")
    df_plot["time_ms"].plot(kind="bar", title="Execution time (ms) - large dataset")
    plt.ylabel("Time (ms)")
    plt.tight_layout()
    plt.savefig(GRAPHS / "time_by_mode.png")
    print("Saved", GRAPHS / "time_by_mode.png")

    # Speedup plot (relative to serial)
    if serial_time is not None:
        df_speed = df[df["mode"] != "serial"].copy()
        df_speed["speedup"] = serial_time / df_speed["time_ms"]
        plt.figure(figsize=(8, 5))
        plt.plot(df_speed["label"], df_speed["speedup"], marker="o")
        plt.title("Speedup vs Serial (large dataset)")
        plt.xlabel("Configuration")
        plt.ylabel("Speedup")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(GRAPHS / "speedup_vs_serial.png")
        print("Saved", GRAPHS / "speedup_vs_serial.png")
    else:
        print("Serial baseline not found; skipping speedup plot")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
