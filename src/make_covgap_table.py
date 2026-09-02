"""Appendix CovGap table from the existing headline results (closes the
'CovGap promised, reported nowhere' gap, appendix.tex STILL-MISSING #1).

CovGap = (1/K) sum_c |cov_c - (1-alpha)| in percentage points, already
computed in every headline results row (no re-run needed). This tabulates it
in the SAME layout as Table 2 (make_headline_tables.py) and, crucially, for
the SAME per-cell configuration Table 2 displays: each baseline is shown at
its best-by-SIZE score x train_frac (not its best CovGap), so the two tables
cross-reference exactly. Bold marks the SMALLEST CovGap per (dataset, budget)
-- the honest 'who has the best class-conditional coverage' marker, which is
often NOT FRCP (the ordinary tension between tight sets and conditional
coverage; e.g. FRCP is worse than CV+ at s=14 on CIFAR-10 / miniImageNet).

Usage (from repo root):
    python src/make_covgap_table.py
"""
import argparse, json, os

from make_headline_tables import (DS_LABEL, FROZEN_ARM, MAIN_DS, best_row)

# same arms as Table 2, but the frozen row is labelled "FRCP" to match the
# paper register (make_headline_tables still emits the legacy "Ours (full CP)")
ARMS = [("frozen", "FRCP"), ("cvplus", "CV+"),
        ("splitcp", "Split CP (best)"), ("semicp", "SemiCP (best)")]


def build(results_dir, datasets, alpha, shots_sel, label, caption, out):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{ll{'c' * len(shots_sel)}}}",
             r"\toprule",
             "dataset & method & "
             + " & ".join(rf"{s} shots" for s in shots_sel) + r"\\"]
    for ds in datasets:
        path = os.path.join(results_dir, f"results_{ds}.json")
        rows = json.load(open(path))["rows"]
        fro = FROZEN_ARM.get(ds, "frozen")
        lines.append(r"\midrule")
        # smallest CovGap per budget among the size-optimal cells of each arm
        best = {}
        for s in shots_sel:
            gaps = [b["covgap"] for a, _ in ARMS
                    if (b := best_row(rows, fro if a == "frozen" else a,
                                      s, alpha))]
            best[s] = min(gaps) if gaps else None
        for i, (arm, arm_label) in enumerate(ARMS):
            a = fro if arm == "frozen" else arm
            cells = []
            for s in shots_sel:
                b = best_row(rows, a, s, alpha)
                if b is None:
                    cells.append("--")
                    continue
                txt = f"{b['covgap']:.1f}"
                if best[s] is not None and b["covgap"] <= best[s] + 1e-6:
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

    base_cap = ("Class-conditional coverage gap "
                "CovGap $=\\frac{1}{K}\\sum_c |\\mathrm{cov}_c - (1-\\alpha)|$ "
                "in percentage points (50 trials, balanced split), for the "
                "same per-cell configuration shown in "
                "Table~\\ref{tab:headline-main}: each baseline at its "
                "best-by-size score and train fraction. Lower is better; "
                "bold marks the smallest CovGap per dataset and budget. FRCP "
                "trades some class-conditional coverage for smaller sets, so "
                "at larger budgets CV+ attains a smaller gap.")
    build(args.results_dir, MAIN_DS, 0.1, args.shots,
          "tab:covgap-main",
          base_cap + " Target miscoverage $\\alpha=0.1$.",
          os.path.join(args.out_dir, "table_covgap_main_a01.tex"))
    build(args.results_dir, MAIN_DS, 0.05, args.shots,
          "tab:covgap-appendix-a005",
          base_cap + " Target miscoverage $\\alpha=0.05$.",
          os.path.join(args.out_dir, "table_covgap_appendix_a005.tex"))


if __name__ == "__main__":
    main()
