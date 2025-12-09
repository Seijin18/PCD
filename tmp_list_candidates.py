from pathlib import Path

ROOT = Path(".").resolve()
DATA = ROOT / "Entrega" / "Data"
data_file = DATA / "dados_small.csv"
try:
    data_len = sum(1 for _ in open(data_file, "r", encoding="utf-8"))
except Exception:
    data_len = None
print("data_len=", data_len)
prefixes = ["openmp", "omp"]
for p in prefixes:
    cands = list(ROOT.glob(f"**/assign_{p}*.csv"))
    print("\nprefix", p, "candidates:")
    for c in cands:
        try:
            cnt = sum(1 for _ in open(c, "r", encoding="utf-8"))
        except Exception:
            cnt = None
        print(c.resolve(), "lines=", cnt)
