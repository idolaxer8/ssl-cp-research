"""Refined-representation baseline table (fair-representation control).

Answers reviewer M2 "is FRCP's win just a better representation?": it hands the
SAME pool-fitted T->W->S refinement to the split-CP competitors and asks whether
FRCP still wins. Rows per dataset = FRCP, then each baseline in raw and refined
form; cell = mean set size $\\pm$ SE with coverage; bold = smallest per
(dataset, budget). Baselines are best-over-score x train_frac per cell, exactly
as Table 2. FRCP + raw baselines come from output/headline; refined baselines
(arms *_ref) from output/headline_refined_baselines.

Usage (from repo root):
    python src/make_refined_baseline_table.py
"""
import argparse, json, os

from make_headline_tables import DS_LABEL, FROZEN_ARM, MAIN_DS, best_row

# (source, arm, label) -- source picks raw_dir vs ref_dir; "__frozen__" resolves
# per dataset (topk NCM on aircraft/cars).
ROWS = [
    ("raw", "__frozen__", "FRCP"),
    ("raw", "cvplus",      "CV+ (raw)"),
    ("ref", "cvplus_ref",  "CV+ (refined)"),
    ("raw", "splitcp",     "Split CP (raw)"),
    ("ref", "splitcp_ref", "Split CP (refined)"),
    ("raw", "semicp",      "SemiCP (raw)"),
    ("ref", "semicp_ref",  "SemiCP (refined)"),
]


def build(raw_dir, ref_dir, datasets, alpha, shots_sel, label, caption, out):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{ll{'c' * len(shots_sel)}}}",
             r"\toprule",
             "dataset & method & "
             + " & ".join(rf"{s} shots" for s in shots_sel) + r"\\"]
    for ds in datasets:
        raw = json.load(open(os.path.join(raw_dir, f"results_{ds}.json")))["rows"]
        ref = json.load(open(os.path.join(ref_dir, f"results_{ds}.json")))["rows"]
        fro = FROZEN_ARM.get(ds, "frozen")
        lines.append(r"\midrule")
        picked = {}
        for src, arm, lbl in ROWS:
            a = fro if arm == "__frozen__" else arm
            rws = raw if src == "raw" else ref
            picked[lbl] = {s: best_row(rws, a, s, alpha) for s in shots_sel}
        best = {}
        for s in shots_sel:
            szs = [picked[lbl][s]["sz"] for _, _, lbl in ROWS if picked[lbl][s]]
            best[s] = min(szs) if szs else None
        for i, (_, _, lbl) in enumerate(ROWS):
            cells = []
            for s in shots_sel:
                b = picked[lbl][s]
                if b is None:
                    cells.append("--")
                    continue
                txt = (f"{b['sz']:.2f}$\\pm${b['sz_se']:.2f} "
                       rf"{{\scriptsize({b['cov']:.3f})}}")
                if best[s] is not None and b["sz"] <= best[s] + 5e-3:
                    txt = rf"\textbf{{{txt}}}"
                cells.append(txt)
            head = (rf"\multirow{{{len(ROWS)}}}{{*}}"
                    rf"{{{DS_LABEL.get(ds, ds)}}}" if i == 0 else "")
            lines.append(f"{head} & {lbl} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default="output/headline")
    ap.add_argument("--ref_dir", default="output/headline_refined_baselines")
    ap.add_argument("--out_dir", default="output/headline_refined_baselines/plots")
    ap.add_argument("--shots", type=int, nargs="+", default=[2, 4, 8, 14])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cap = ("Fair-representation control: mean prediction-set size ($\\pm$ SE, "
           "50 trials, balanced split) with marginal coverage in parentheses, "
           "when the split-CP baselines are given the SAME pool-fitted "
           "refinement as FRCP. Baselines are shown at their best score and "
           "train fraction per cell. Bold marks the smallest set per dataset "
           "and budget. Below the boundary ($s=2$) FRCP stays smallest even "
           "against refined competitors, so its advantage there is not merely "
           "the representation; refinement helps the trained-head baselines at "
           "$K=100$ but not at $K=10$, where it is fitted for the prototype "
           "score.")
    build(args.raw_dir, args.ref_dir, MAIN_DS, 0.1, args.shots,
          "tab:refined-baselines",
          cap + " Target miscoverage $\\alpha=0.1$.",
          os.path.join(args.out_dir, "table_refined_baselines_a01.tex"))
    build(args.raw_dir, args.ref_dir, MAIN_DS, 0.05, args.shots,
          "tab:refined-baselines-a005",
          cap + " Target miscoverage $\\alpha=0.05$.",
          os.path.join(args.out_dir, "table_refined_baselines_a005.tex"))


if __name__ == "__main__":
    main()
