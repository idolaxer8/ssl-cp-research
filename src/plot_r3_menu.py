"""Paper Table 4: transform-menu ablation (R3) booktabs table per dataset.

Rows = transform arms (raw / T only / T+W / T+W+D full / W only full-rank),
column groups = NCM (champion prototype-softmax, geodesic top-k), sub-
columns = calibration sizes. Bold = best per column (ties at display
precision). From transform_control_experiment.py results.

Usage:
    python src/plot_r3_menu.py --results output/r3_menu/results_cifar100.json
"""
import argparse, json, os

DS_LABEL = {"cifar100": "CIFAR-100", "miniimagenet": "miniImageNet",
            "cub200": "CUB-200", "food101": "Food-101",
            "aircraft": "FGVC-Aircraft", "stanford_cars": "Stanford Cars"}
ARM_ORDER = ["raw768", "pca128", "pca128_lwcw", "qe_pca128_lwcw",
             "lw_cluster768"]
ARM_TEX = {"raw768": "raw embedding",
           "pca128": "T only",
           "pca128_lwcw": "T$+$W",
           "qe_pca128_lwcw": "T$+$W$+$D (full)",
           "lw_cluster768": "W only (full rank)"}
NCM_ORDER = ["prototype_softmax", "unwhitened_topk_asym"]
NCM_TEX = {"prototype_softmax": "prototype-softmax",
           "unwhitened_topk_asym": "geodesic top-$k$"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out_dir", default="output/r3_menu/plots")
    ap.add_argument("--split", default="balanced_both")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for path in args.results:
        r = json.load(open(path))
        ds = r["dataset"]
        rows = [x for x in r["rows"] if x["split"] == args.split]
        cals = sorted({x["cal"] for x in rows})
        trials = r["config"]["n_trials"]

        def cell(arm, ncm, cal):
            c = [x for x in rows if x["arm"] == arm and x["ncm"] == ncm
                 and x["cal"] == cal]
            return c[0] if c else None

        best = {}
        for ncm in NCM_ORDER:
            for cal in cals:
                vals = [cell(a, ncm, cal) for a in ARM_ORDER]
                vals = [v["sz"] for v in vals if v]
                best[(ncm, cal)] = min(vals) if vals else None

        ncol = len(cals) * len(NCM_ORDER)
        lines = [r"\begin{table}[t]", r"\centering", r"\small",
                 rf"\caption{{Transform-menu ablation on "
                 rf"{DS_LABEL.get(ds, ds)}: mean prediction-set size "
                 rf"($\pm$ SE, {trials} trials, balanced split, "
                 r"$\alpha=0.1$). Every arm is a pool-fit transform under "
                 r"exact full CP. Bold: best per column.}",
                 rf"\label{{tab:r3-menu-{ds}}}",
                 rf"\begin{{tabular}}{{l{'c' * ncol}}}", r"\toprule"]
        span = " & ".join(
            rf"\multicolumn{{{len(cals)}}}{{c}}{{{NCM_TEX[n]}}}"
            for n in NCM_ORDER)
        lines.append(rf"transform & {span}\\")
        lines.append("$n_{\\mathrm{cal}}$ & "
                     + " & ".join(str(c) for _ in NCM_ORDER for c in cals)
                     + r"\\")
        lines.append(r"\midrule")
        for a in ARM_ORDER:
            cells = []
            for n in NCM_ORDER:
                for cal in cals:
                    b = cell(a, n, cal)
                    if not b:
                        cells.append("--")
                        continue
                    txt = f"{b['sz']:.2f}$\\pm${b['sz_se']:.2f}"
                    if b["sz"] <= best[(n, cal)] + 5e-3:
                        txt = rf"\textbf{{{txt}}}"
                    cells.append(txt)
            lines.append(f"{ARM_TEX[a]} & " + " & ".join(cells) + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        out = os.path.join(args.out_dir, f"table_r3_{ds}.tex")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"saved {out}")
        # console summary
        print(f"{ds} (balanced, alpha 0.1, {trials} trials):")
        for a in ARM_ORDER:
            line = f"  {ARM_TEX[a]:<22}"
            for n in NCM_ORDER:
                for cal in cals:
                    b = cell(a, n, cal)
                    line += f" {b['sz']:7.2f}" if b else "      --"
            print(line)


if __name__ == "__main__":
    main()
