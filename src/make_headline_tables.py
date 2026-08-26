"""Paper Table 2 (+ appendix / internal companions) from headline results.

Layout: rows grouped by dataset (4 method rows each), columns = shot
budgets; cell = mean set size $\\pm$ SE with coverage in scriptsize
parens; bold = smallest size per (dataset, budget). Baselines follow the
table convention: best over score x train_frac per cell. The frozen row
uses the per-regime champion NCM (prototype-softmax; geodesic top-k on
fine-grained, where the softmax auto-T collapse is documented).

Split (user call 2026-08-26):
    main      cifar10 cifar100 miniimagenet eurosat aircraft @ alpha 0.1
    appendix  same datasets @ alpha 0.05
    internal  cub200 food101 stanford_cars (viewing only, both alphas)

Usage (from repo root):
    python src/make_headline_tables.py
"""
import argparse, json, os

DS_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
            "miniimagenet": "miniImageNet", "eurosat": "EuroSAT",
            "cub200": "CUB-200", "food101": "Food-101",
            "aircraft": "FGVC-Aircraft", "stanford_cars": "Stanford Cars"}
FROZEN_ARM = {"aircraft": "frozen_unwhitened_topk_asym",
              "stanford_cars": "frozen_unwhitened_topk_asym"}
ARMS = [("frozen", "Ours (full CP)"), ("cvplus", "CV+"),
        ("splitcp", "Split CP (best)"), ("semicp", "SemiCP (best)")]

MAIN_DS = ["cifar10", "cifar100", "miniimagenet", "eurosat", "aircraft"]
INTERNAL_DS = ["cub200", "food101", "stanford_cars"]


def best_row(rows, arm, shots, alpha):
    cand = [x for x in rows if x["arm"] == arm and x["shots"] == shots
            and x["alpha"] == alpha]
    return min(cand, key=lambda x: x["sz"]) if cand else None


def build(results_dir, datasets, alpha, shots_sel, label, caption, out):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{ll{'c' * len(shots_sel)}}}",
             r"\toprule",
             "dataset & method & "
             + " & ".join(rf"{s} shots" for s in shots_sel) + r"\\"]
    for ds in datasets:
        path = os.path.join(results_dir, f"results_{ds}.json")
        r = json.load(open(path))
        rows = r["rows"]
        fro = FROZEN_ARM.get(ds, "frozen")
        lines.append(r"\midrule")
        best = {s: min(v["sz"] for a, _ in ARMS
                       if (v := best_row(rows, fro if a == "frozen" else a,
                                         s, alpha)))
                for s in shots_sel}
        for i, (arm, arm_label) in enumerate(ARMS):
            a = fro if arm == "frozen" else arm
            cells = []
            for s in shots_sel:
                b = best_row(rows, a, s, alpha)
                if b is None:
                    cells.append("--")
                    continue
                txt = (f"{b['sz']:.2f}$\\pm${b['sz_se']:.2f} "
                       rf"{{\scriptsize({b['cov']:.3f})}}")
                if b["sz"] <= best[s] + 5e-3:
                    txt = rf"\textbf{{{txt}}}"
                cells.append(txt)
            head = (rf"\multirow{{{len(ARMS)}}}{{*}}"
                    rf"{{{DS_LABEL.get(ds, ds)}}}" if i == 0 else "")
            lines.append(f"{head} & {arm_label} & " + " & ".join(cells)
                         + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="output/headline")
    ap.add_argument("--out_dir", default="output/headline/plots")
    ap.add_argument("--shots", type=int, nargs="+", default=[2, 4, 8, 14])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base_cap = ("Headline comparison: mean prediction-set size ($\\pm$ SE, "
                "50 trials, balanced split) with marginal coverage in "
                "parentheses. Split CP and SemiCP are shown at their best "
                "score (THR/APS/RAPS) and train fraction per cell. Ours = "
                "the frozen pool-fitted refinement with exact full CP "
                "(geodesic top-$k$ NCM on FGVC-Aircraft). Bold: smallest "
                "set per dataset and budget.")
    build(args.results_dir, MAIN_DS, 0.1, args.shots,
          "tab:headline-main",
          base_cap + " Target miscoverage $\\alpha=0.1$.",
          os.path.join(args.out_dir, "table_headline_main_a01.tex"))
    build(args.results_dir, MAIN_DS, 0.05, args.shots,
          "tab:headline-appendix-a005",
          base_cap + " Target miscoverage $\\alpha=0.05$ (appendix "
          "companion to Table~\\ref{tab:headline-main}).",
          os.path.join(args.out_dir, "table_headline_appendix_a005.tex"))
    for alpha, tag in ((0.1, "a01"), (0.05, "a005")):
        build(args.results_dir, INTERNAL_DS, alpha, args.shots,
              f"tab:headline-internal-{tag}",
              f"INTERNAL (not for submission): remaining datasets at "
              f"$\\alpha={alpha:g}$.",
              os.path.join(args.out_dir,
                           f"table_headline_internal_{tag}.tex"))


if __name__ == "__main__":
    main()
