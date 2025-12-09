from pathlib import Path

ROOT = Path(".").resolve()
RESULTS = ROOT / "Entrega" / "results"
DATA = ROOT / "Entrega" / "Data"
data_file = DATA / "dados_small.csv"
try:
    data_len = sum(1 for _ in open(data_file, "r", encoding="utf-8"))
except Exception:
    data_len = None
print("data_len=", data_len)

cands = list(ROOT.glob("**/assign_omp_*.csv"))
print("candidates count", len(cands))
for c in cands:
    print(" -", c, sum(1 for _ in open(c, "r", encoding="utf-8")))


def choose_best(cands):
    if not cands:
        return None
    # if we know data length, pick the candidate with line count closest to it
    if data_len is not None:
        best = None
        best_diff = None
        for c in cands:
            try:
                cnt = sum(1 for _ in open(c, "r", encoding="utf-8"))
            except Exception:
                continue
            # ignore obviously huge legacy files (100x larger than expected)
            if cnt > data_len * 100:
                continue
            diff = abs(cnt - data_len)
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


print("choose_best ->", choose_best(cands))
