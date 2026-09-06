"""Paper headline tables from the frozen-champion comparison runs.

Post sec5 reorg (user call 2026-09-06): headline datasets = CIFAR-10 /
CIFAR-100 / EuroSAT (miniImageNet dropped from the comparison entirely),
max tabulated budget = 12 shots. DINOv2 ViT-B carries the main-text
table; CLIP ViT-B moves to the appendix; CLIP ViT-L emitted for the
large-encoder comparison. One table per backbone x alpha. Rows grouped
by dataset; within each block the baselines come first and FRCP (ours)
is the LAST row (standard comparison convention). Cell = mean set size
$\\pm$ SE with coverage in scriptsize parens; bold = smallest size per
(dataset, budget). Baselines follow the strongest-baseline convention:
best over score x train_frac per cell.

The requested --shots grid is intersected per results dir with the shots
actually present (the CLIP runs currently stop at {2,4,8,14}); missing
budgets are dropped with a warning rather than emitted as empty columns.

Outputs (out_dir):
    table_headline_dinov2_a01.tex    main Table 2 (alpha = 0.1)
    table_headline_dinov2_a005.tex   appendix companion (alpha = 0.05)
    table_headline_clipb_a01.tex     CLIP-B appendix table
    table_headline_clipb_a005.tex    appendix companion
    table_headline_clipl_a01.tex     CLIP-L large-encoder table
    table_headline_clipl_a005.tex    appendix companion
    table_headline_internal_*.tex    dropped datasets (viewing only)

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
# baselines first, ours last (comparison-table convention, user call 09-02)
ARMS = [("splitcp", "Split CP (best)"), ("cvplus", "CV+"),
        ("semicp", "SemiCP (best)"), ("frozen", "FRCP (ours)")]

MAIN_DS = ["cifar10", "cifar100", "eurosat"]
INTERNAL_DS = ["miniimagenet", "aircraft", "cub200", "food101",
               "stanford_cars"]


def best_row(rows, arm, shots, alpha):
    cand = [x for x in rows if x["arm"] == arm and x["shots"] == shots
            and x["alpha"] == alpha]
    return min(cand, key=lambda x: x["sz"]) if cand else None


def available_shots(results_dir, datasets, shots_req):
    """Intersect the requested grid with the shots present in every
    results file of this dir (missing budgets are dropped, not tabulated
    as empty columns)."""
    have = None
    for ds in datasets:
        path = os.path.join(results_dir, f"results_{ds}.json")
        if not os.path.exists(path):
            continue
        s = {r["shots"] for r in json.load(open(path))["rows"]}
        have = s if have is None else have & s
    if have is None:
        return shots_req
    kept = [s for s in shots_req if s in have]
    if kept != shots_req:
        print(f"[warn] {results_dir}: shots {sorted(set(shots_req) - have)}"
              f" absent, tabulating {kept}")
    return kept


def build(results_dir, datasets, alpha, shots_sel, label, caption, out):
    lines = [r"\begin{table}[t]", r"\centering", r"\small",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{ll{'c' * len(shots_sel)}}}",
             r"\toprule",
             "dataset & method & "
             + " & ".join(rf"{s} shots" for s in shots_sel) + r"\\"]
    for ds in datasets:
        path = os.path.join(results_dir, f"results_{ds}.json")
        if not os.path.exists(path):
            print(f"[skip] {path} missing")
            continue
        rows = json.load(open(path))["rows"]
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
    ap.add_argument("--headline_dir", default="output/headline",
                    help="dinov2 results (the Table 2 run)")
    ap.add_argument("--clipb_dir",
                    default="output/backbone_headline/clip-base")
    ap.add_argument("--clipl_dir",
                    default="output/backbone_headline/clip-large")
    ap.add_argument("--out_dir", default="output/headline/plots")
    ap.add_argument("--shots", type=int, nargs="+", default=[2, 4, 8, 12],
                    help="requested grid, intersected per results dir")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    base_cap = ("mean prediction-set size ($\\pm$ SE, 50 trials, balanced "
                "split) with marginal coverage in parentheses. Split CP and "
                "SemiCP are shown at their best score (THR/APS/RAPS) and "
                "train fraction per cell (strongest-baseline convention). "
                "FRCP = the frozen pool-fitted refinement with exact full "
                "CP. Bold: smallest set per dataset and budget.")
    backbones = [
        ("dinov2", "DINOv2 ViT-B", args.headline_dir,
         {0.1: "tab:headline-main", 0.05: "tab:headline-appendix-a005"}),
        ("clipb", "CLIP ViT-B", args.clipb_dir,
         {0.1: "tab:headline-clipb", 0.05: "tab:headline-clipb-a005"}),
        ("clipl", "CLIP ViT-L", args.clipl_dir,
         {0.1: "tab:headline-clipl", 0.05: "tab:headline-clipl-a005"}),
    ]
    for bb_tag, bb_name, res_dir, lab in backbones:
        shots_sel = available_shots(res_dir, MAIN_DS, args.shots)
        for alpha, atag in ((0.1, "a01"), (0.05, "a005")):
            build(res_dir, MAIN_DS, alpha, shots_sel, lab[alpha],
                  f"Headline comparison on {bb_name} embeddings: "
                  + base_cap + f" Target miscoverage $\\alpha={alpha:g}$.",
                  os.path.join(args.out_dir,
                               f"table_headline_{bb_tag}_{atag}.tex"))
    shots_sel = available_shots(args.headline_dir, INTERNAL_DS, args.shots)
    for alpha, atag in ((0.1, "a01"), (0.05, "a005")):
        build(args.headline_dir, INTERNAL_DS, alpha, shots_sel,
              f"tab:headline-internal-{atag}",
              f"INTERNAL (not for submission): datasets outside the "
              f"09-06 scope at $\\alpha={alpha:g}$.",
              os.path.join(args.out_dir,
                           f"table_headline_internal_{atag}.tex"))


if __name__ == "__main__":
    main()
