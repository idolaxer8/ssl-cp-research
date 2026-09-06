"""Runtime table from headline_timing_benchmark.py output.

Sec 5 reorg (user call 2026-09-06): runtime is a main-text result. One
booktabs table per timing_<ds>.json: rows = methods (baselines first,
FRCP last, headline-table convention), columns = shot budgets, cell =
median per-trial seconds over the reps (median absorbs the GPU warm-up
rep). FRCP's one-off pool-transform fit + temperature pilot go in the
caption, since they amortize over every trial and deployment.

Usage (from repo root):
    python src/make_runtime_table.py --timing_dir output/headline \
        --datasets cifar100 cifar10 eurosat --out_dir output/headline/plots
"""
import argparse, json, os, statistics

DS_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
            "miniimagenet": "miniImageNet", "eurosat": "EuroSAT",
            "stanford_cars": "Stanford Cars"}
ARMS = [("splitcp", "Split CP"), ("cvplus", "CV+"),
        ("semicp", "SemiCP"), ("frozen", "FRCP (ours)")]


def med(rows, arm, s):
    v = [r["seconds"] for r in rows if r["arm"] == arm and r["shots"] == s]
    return statistics.median(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing_dir", default="output/headline")
    ap.add_argument("--datasets", nargs="+",
                    default=["cifar100", "cifar10", "eurosat"])
    ap.add_argument("--out_dir", default="output/headline/plots")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    loaded = {}
    for ds in args.datasets:
        p = os.path.join(args.timing_dir, f"timing_{ds}.json")
        if not os.path.exists(p):
            print(f"[skip] {p} missing")
            continue
        loaded[ds] = json.load(open(p))

    # one combined table: dataset blocks, methods x shots
    shots = sorted({r["shots"] for d in loaded.values() for r in d["rows"]})
    for ds, d in loaded.items():
        n_reps = d["config"]["n_reps"]
        one = d["one_off"]
        lines = [r"\begin{table}[t]", r"\centering", r"\small",
                 r"\setlength{\tabcolsep}{5pt}"]
        cap = (f"Per-trial compute time in seconds on {DS_LABEL[ds]} "
               f"($K={d['K']}$, {d['n_test']} test points, single GPU, "
               f"median over {n_reps} trials), timed standalone on the "
               "headline protocol at target grid budgets. Split-based "
               "rows include their probe fit and all three scores; CV+ "
               "includes its per-fold probe fits. FRCP additionally pays "
               "a one-off pool-transform fit of "
               f"{one['transform_fit_s']:.1f}\\,s"
               + (f" and a {one['T_pilot_s']:.1f}\\,s temperature pilot"
                  if "T_pilot_s" in one else "")
               + ", shared by every trial, budget and test batch.")
        lines += [rf"\caption{{{cap}}}",
                  rf"\label{{tab:runtime-{ds}}}",
                  rf"\begin{{tabular}}{{l{'c' * len(shots)}}}",
                  r"\toprule",
                  "method & " + " & ".join(rf"{s} shots" for s in shots)
                  + r"\\", r"\midrule"]
        for arm, label in ARMS:
            cells = []
            for s in shots:
                m = med(d["rows"], arm, s)
                cells.append("--" if m is None else f"{m:.2f}")
            lines.append(f"{label} & " + " & ".join(cells) + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        out = os.path.join(args.out_dir, f"table_runtime_{ds}.tex")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"saved {out}")

    # prose-ready summary
    for ds, d in loaded.items():
        print(f"--- {ds} (K={d['K']}, n_test={d['n_test']}, "
              f"one_off={ {k: round(v, 1) for k, v in d['one_off'].items()} })")
        for arm, label in ARMS:
            line = "  " + f"{label:>12}: "
            for s in shots:
                m = med(d["rows"], arm, s)
                line += f" s{s}={m:.2f}" if m is not None else f" s{s}=--"
            print(line)


if __name__ == "__main__":
    main()
