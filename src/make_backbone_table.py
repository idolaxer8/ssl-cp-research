"""Section 5.5 backbone-transfer table: FRCP vs the strongest baseline on each
backbone x dataset, all measured with the exact Table 2 pipeline and protocol
(headline_experiment.py --emb_suffix, 50 trials, shots {2,4,8,14}).

Cell = "FRCP / best baseline" mean set size; bold on the smaller. The best
baseline is min over CV+ / split CP / SemiCP at their best score x train_frac
(strongest-baseline convention, as in Table 2). Coverage is not printed per
cell; the caption reports the FRCP coverage range (valid everywhere).

Sources: dinov2 anchor rows from output/headline/results_<ds>.json (the
Table 2 run); other backbones from output/backbone_headline/<bb>/. FRCP arm
per cell follows the champion assignment: prototype softmax, geodesic top-k
on FGVC-Aircraft, and prototype cosine on ssl-resnet50/CIFAR-100 where the
pilot temperature fit collapses (T -> 0.0007, documented).

Usage (from repo root):
    python src/make_backbone_table.py
"""
import argparse, json, os

DS = ["cifar10", "cifar100", "miniimagenet", "eurosat", "aircraft"]
DS_LABEL = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100",
            "miniimagenet": "miniImageNet", "eurosat": "EuroSAT",
            "aircraft": "FGVC-Aircraft"}
BB = ["dinov2", "clip-base", "clip-large", "ssl-resnet50"]
BB_LABEL = {"dinov2": "DINOv2-B", "clip-base": "CLIP-B",
            "clip-large": "CLIP-L", "ssl-resnet50": "SSL-ResNet50"}
BASELINES = ["cvplus", "splitcp", "semicp"]
SHOTS = [2, 4, 8, 14]


def frozen_arm(bb, ds):
    if ds == "aircraft":
        return "frozen_unwhitened_topk_asym"
    if bb == "ssl-resnet50" and ds == "cifar100":
        return "frozen_prototype_cosine"   # auto-T collapse fallback
    return "frozen"


def best_row(rows, arm, s, alpha):
    c = [x for x in rows if x["arm"] == arm and x["shots"] == s
         and x["alpha"] == alpha]
    return min(c, key=lambda x: x["sz"]) if c else None


def load_rows(headline_dir, backbone_dir, bb, ds):
    d = headline_dir if bb == "dinov2" else os.path.join(backbone_dir, bb)
    p = os.path.join(d, f"results_{ds}.json")
    return json.load(open(p))["rows"] if os.path.exists(p) else None


def build(headline_dir, backbone_dir, alpha, label, caption, out):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{ll{'c' * len(SHOTS)}}}",
             r"\toprule",
             "dataset & backbone & "
             + " & ".join(rf"{s} shots" for s in SHOTS) + r"\\"]
    covs = []
    for ds in DS:
        lines.append(r"\midrule")
        first = True
        for bb in BB:
            rows = load_rows(headline_dir, backbone_dir, bb, ds)
            if rows is None:
                cells = ["--"] * len(SHOTS)
            else:
                cells = []
                for s in SHOTS:
                    f = best_row(rows, frozen_arm(bb, ds), s, alpha)
                    b = min((v for a in BASELINES
                             if (v := best_row(rows, a, s, alpha))),
                            key=lambda x: x["sz"], default=None)
                    if f is None or b is None:
                        cells.append("--")
                        continue
                    covs.append(f["cov"])
                    ft, bt = f"{f['sz']:.2f}", f"{b['sz']:.2f}"
                    if f["sz"] <= b["sz"] + 5e-3:
                        ft = rf"\textbf{{{ft}}}"
                    else:
                        bt = rf"\textbf{{{bt}}}"
                    cells.append(f"{ft} / {bt}")
            head = (rf"\multirow{{{len(BB)}}}{{*}}{{{DS_LABEL[ds]}}}"
                    if first else "")
            first = False
            lines.append(f"{head} & {BB_LABEL[bb]} & "
                         + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {out}  (FRCP coverage {min(covs):.3f}-{max(covs):.3f} "
          f"over {len(covs)} cells)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headline_dir", default="output/headline")
    ap.add_argument("--backbone_dir", default="output/backbone_headline")
    ap.add_argument("--out_dir", default="output/backbone_headline/plots")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cap = ("Backbone transfer under the full pipeline and protocol of "
           "Table~\\ref{tab:headline-main}: mean prediction-set size for "
           "FRCP~/~the strongest baseline per cell (best of CV+, split CP, "
           "SemiCP at their best score and train fraction; 50 trials, "
           "balanced split). Bold marks the smaller of the pair. FRCP uses "
           "the prototype softmax score, the top-$k$ ratio on FGVC-Aircraft, "
           "and the prototype cosine score on SSL-ResNet50/CIFAR-100, where "
           "the pilot temperature fit collapses. FRCP holds coverage in "
           "every cell.")
    build(args.headline_dir, args.backbone_dir, 0.1,
          "tab:backbone-transfer",
          cap + " Target miscoverage $\\alpha=0.1$.",
          os.path.join(args.out_dir, "table_backbone_transfer_a01.tex"))
    build(args.headline_dir, args.backbone_dir, 0.05,
          "tab:backbone-transfer-a005",
          cap + " Target miscoverage $\\alpha=0.05$.",
          os.path.join(args.out_dir, "table_backbone_transfer_a005.tex"))


if __name__ == "__main__":
    main()
