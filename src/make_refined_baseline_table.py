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
# per dataset (topk NCM on aircraft/cars). Baselines first, ours last
# (comparison convention, user call 09-02).
ROWS = [
    ("raw", "splitcp",     "Split CP (raw)"),
    ("ref", "splitcp_ref", "Split CP (refined)"),
    ("raw", "cvplus",      "CV+ (raw)"),
    ("ref", "cvplus_ref",  "CV+ (refined)"),
    ("raw", "semicp",      "SemiCP (raw)"),
    ("ref", "semicp_ref",  "SemiCP (refined)"),
    ("raw", "__frozen__",  "FRCP (ours)"),
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
    ap.add_argument("--clipb_raw_dir",
                    default="output/backbone_headline/clip-base")
    ap.add_argument("--clipb_ref_dir",
                    default="output/headline_refined_baselines/clip-base")
    ap.add_argument("--out_dir", default="output/headline_refined_baselines/plots")
    ap.add_argument("--shots", type=int, nargs="+", default=[2, 4, 8, 14])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cap = ("Fair-representation control on {bb} embeddings: mean "
           "prediction-set size ($\\pm$ SE, 50 trials, balanced split) with "
           "marginal coverage in parentheses, when the split-CP baselines "
           "are handed the same pool-fitted refinement FRCP uses. Baselines "
           "are shown at their best score and train fraction per cell. Bold "
           "marks the smallest set per dataset and budget.")
    for bb_tag, bb_name, raw_d, ref_d in [
            ("dinov2", "DINOv2 ViT-B", args.raw_dir, args.ref_dir),
            ("clipb", "CLIP ViT-B", args.clipb_raw_dir, args.clipb_ref_dir)]:
        if not os.path.exists(os.path.join(ref_d,
                                           f"results_{MAIN_DS[0]}.json")):
            print(f"[skip] {bb_tag}: no refined results in {ref_d}")
            continue
        for alpha, atag in ((0.1, "a01"), (0.05, "a005")):
            build(raw_d, ref_d, MAIN_DS, alpha, args.shots,
                  f"tab:refined-baselines-{bb_tag}-{atag}",
                  cap.format(bb=bb_name)
                  + f" Target miscoverage $\\alpha={alpha:g}$.",
                  os.path.join(
                      args.out_dir,
                      f"table_refined_baselines_{bb_tag}_{atag}.tex"))


if __name__ == "__main__":
    main()
