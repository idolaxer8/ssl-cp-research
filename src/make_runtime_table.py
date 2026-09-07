"""Runtime table from the timing benchmarks.

--mode trial (09-06): per-trial seconds from timing_<ds>.json
(headline_timing_benchmark.py; probe fits and calibration included).
--mode testtime (09-07 final): per-test-batch seconds from
timing_testtime_<ds>.json (testtime_benchmark.py; only the phase that
runs when a new test batch arrives, everything else prepared), with the
FRCP (naive) row = the unvectorized CPU transductive loop.

Rows = methods (baselines first, FRCP last, headline-table convention),
columns = shot budgets, cell = median seconds over the reps. One-off /
calibration-side costs go in the caption.

Usage (from repo root):
    python src/make_runtime_table.py --mode testtime \
        --timing_dir output/headline --datasets cifar100 cifar10 eurosat
"""
import argparse, json, os, statistics

DS_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
            "miniimagenet": "miniImageNet", "eurosat": "EuroSAT",
            "stanford_cars": "Stanford Cars"}
ARMS = [("splitcp", "Split CP"), ("cvplus", "CV+"),
        ("semicp", "SemiCP"), ("frozen", "FRCP (ours)")]
ARMS_TESTTIME = [("splitcp", "Split CP"), ("cvplus", "CV+"),
                 ("semicp", "SemiCP"),
                 ("frozen_naive", "full CP, naive sweep"),
                 ("frozen", "FRCP (ours)")]


def med(rows, arm, s, phase=None):
    v = [r["seconds"] for r in rows if r["arm"] == arm and r["shots"] == s
         and (phase is None or r.get("phase") == phase)]
    return statistics.median(v) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing_dir", default="output/headline")
    ap.add_argument("--datasets", nargs="+",
                    default=["cifar100", "cifar10", "eurosat"])
    ap.add_argument("--out_dir", default="output/headline/plots")
    ap.add_argument("--mode", default="trial",
                    choices=["trial", "testtime"])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tt = args.mode == "testtime"
    arms = ARMS_TESTTIME if tt else ARMS
    prefix = "timing_testtime_" if tt else "timing_"
    phase = "test" if tt else None

    loaded = {}
    for ds in args.datasets:
        p = os.path.join(args.timing_dir, f"{prefix}{ds}.json")
        if not os.path.exists(p):
            print(f"[skip] {p} missing")
            continue
        loaded[ds] = json.load(open(p))

    # one table per dataset: methods x shots
    shots = sorted({r["shots"] for d in loaded.values() for r in d["rows"]})
    for ds, d in loaded.items():
        n_reps = d["config"]["n_reps"]
        one = d["one_off"]
        lines = [r"\begin{table}[t]", r"\centering", r"\small",
                 r"\setlength{\tabcolsep}{5pt}"]
        if tt:
            calib = {label: med(d["rows"], arm, max(shots), "calib")
                     for arm, label in arms if arm != "frozen_naive"}
            calib_txt = ", ".join(
                f"{lbl} {v:.2f}\\,s" for lbl, v in calib.items()
                if v is not None)
            cap = (f"Test-time compute in seconds on {DS_LABEL[ds]} "
                   f"($K={d['K']}$, {d['n_test']} test points, median "
                   f"over {n_reps} trials): only the phase that runs "
                   "when a new test batch arrives. Probe and fold fits, "
                   "pool scoring, calibration quantiles and the "
                   "conformal calibration pass are prepared beforehand "
                   f"(at the largest budget: {calib_txt}; one-off "
                   f"pool-transform fit {one['transform_fit_s']:.1f}\\,s"
                   ", shared by every batch). The naive sweep is the "
                   "same predictor and identical p-values, computed by "
                   "the unvectorized per-candidate transductive loop "
                   "on CPU.")
        else:
            cap = (f"Per-trial compute time in seconds on {DS_LABEL[ds]} "
                   f"($K={d['K']}$, {d['n_test']} test points, single "
                   f"GPU, median over {n_reps} trials), timed standalone "
                   "on the headline protocol at target grid budgets. "
                   "Split-based rows include their probe fit and all "
                   "three scores; CV+ includes its per-fold probe fits. "
                   "FRCP additionally pays a one-off pool-transform fit "
                   f"of {one['transform_fit_s']:.1f}\\,s"
                   + (f" and a {one['T_pilot_s']:.1f}\\,s temperature "
                      "pilot" if "T_pilot_s" in one else "")
                   + ", shared by every trial, budget and test batch.")
        lines += [rf"\caption{{{cap}}}",
                  rf"\label{{tab:runtime-{ds}}}",
                  rf"\begin{{tabular}}{{l{'c' * len(shots)}}}",
                  r"\toprule",
                  "method & " + " & ".join(rf"{s} shots" for s in shots)
                  + r"\\", r"\midrule"]
        for arm, label in arms:
            cells = []
            for s in shots:
                m = med(d["rows"], arm, s, phase)
                cells.append("--" if m is None else f"{m:.2f}")
            lines.append(f"{label} & " + " & ".join(cells) + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        stem = f"table_runtime_testtime_{ds}" if tt else f"table_runtime_{ds}"
        out = os.path.join(args.out_dir, stem + ".tex")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"saved {out}")

    # prose-ready summary
    for ds, d in loaded.items():
        print(f"--- {ds} (K={d['K']}, n_test={d['n_test']}, "
              f"one_off={ {k: round(v, 1) for k, v in d['one_off'].items()} })")
        for arm, label in arms:
            line = "  " + f"{label:>20}: "
            for s in shots:
                m = med(d["rows"], arm, s, phase)
                line += f" s{s}={m:.3f}" if m is not None else f" s{s}=--"
            print(line)
        if tt:
            line = "  " + f"{'calib (frozen)':>20}: "
            for s in shots:
                m = med(d["rows"], "frozen", s, "calib")
                line += f" s{s}={m:.3f}" if m is not None else f" s{s}=--"
            print(line)


if __name__ == "__main__":
    main()
