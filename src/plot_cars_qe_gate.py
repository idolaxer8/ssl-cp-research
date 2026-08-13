"""Complementary plots for the registered cars qe-gate experiment.

Reads output/cars_qe_gate/cars_qe_gate.json (from cars_qe_gate_experiment.py)
and produces, in the same directory:

  cars_qe_gate_verdict.png   set size vs cal budget (wt vs qe_wt vs raw, error
                             bars over trials) + coverage panel + paired-delta
                             panel with the two competing predictions labeled
  cars_qe_gate_dprime.png    empirical per-pair d'-ratio distribution vs the
                             D3 point prediction 1.52 and break-even 1.0
  dprime_predicted_vs_measured.png
                             (if measure_dprime_all.py output is present)
                             (I)-model predicted vs measured d'-ratio for all
                             five datasets -- localizes the idealization error

Linear axes throughout (user preference).
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARM_LABEL = {"raw": "raw (no W/T, no qe)",
             "wt": "champion W/T (no qe)",
             "qe_wt": "DWT = qe + W/T"}
ARM_COLOR = {"raw": "0.55", "wt": "#4c72b0", "qe_wt": "#dd8452"}


def fig_verdict(res, out_dir):
    cals = [int(c) for c in res["cal_sizes"]]
    fig, (ax_sz, ax_cov, ax_d) = plt.subplots(1, 3, figsize=(13.5, 4.0))

    for arm in ["raw", "wt", "qe_wt"]:
        mu = [res["arms"][arm][str(c)]["size_mean"] for c in cals]
        sd = [res["arms"][arm][str(c)]["size_sd"] for c in cals]
        ax_sz.errorbar(cals, mu, yerr=sd, marker="o", capsize=3,
                       color=ARM_COLOR[arm], label=ARM_LABEL[arm])
        cov = [res["arms"][arm][str(c)]["coverage_mean"] for c in cals]
        csd = [res["arms"][arm][str(c)]["coverage_sd"] for c in cals]
        ax_cov.errorbar(cals, cov, yerr=csd, marker="o", capsize=3,
                        color=ARM_COLOR[arm], label=ARM_LABEL[arm])
    ax_sz.set_xlabel("calibration size")
    ax_sz.set_ylabel("avg prediction-set size")
    ax_sz.set_title("Set size (mean ± sd over trials)")
    ax_sz.legend(fontsize=8)
    ax_sz.set_xticks(cals)

    ax_cov.axhline(0.9, color="k", ls=":", lw=1, label="target 1$-\\alpha$")
    ax_cov.set_xlabel("calibration size")
    ax_cov.set_ylabel("coverage")
    ax_cov.set_title("Coverage (validity is free — Prop 1)")
    ax_cov.set_ylim(0.85, 0.97)
    ax_cov.legend(fontsize=8)
    ax_cov.set_xticks(cals)

    # paired delta panel: qe_wt - wt per cal budget
    deltas = [res["verdict"][str(c)]["paired_delta_mean"] for c in cals]
    ses = [res["verdict"][str(c)]["paired_delta_se"] for c in cals]
    rel = [100 * res["verdict"][str(c)]["relative_change_mean"] for c in cals]
    xpos = np.arange(len(cals))
    bars = ax_d.bar(xpos, deltas, yerr=ses, capsize=4,
                    color=["tab:green" if d < 0 else "tab:red" for d in deltas])
    for b, r in zip(bars, rel):
        va = "top" if b.get_height() < 0 else "bottom"
        off = -0.02 if b.get_height() < 0 else 0.02
        ax_d.text(b.get_x() + b.get_width() / 2, b.get_height() + off,
                  f"{r:+.1f}%", ha="center", va=va, fontsize=9)
    ax_d.axhline(0, color="k", lw=1)
    ax_d.set_xticks(xpos)
    ax_d.set_xticklabels([str(c) for c in cals])
    ax_d.set_xlabel("calibration size")
    ax_d.set_ylabel("paired $\\Delta$ set size  (qe$+$W/T $-$ W/T)")
    ax_d.set_title("The registered cell: D3 predicted GAIN ($\\Delta<0$),\n"
                   "folklore-0.7 predicted harm ($\\Delta>0$)")

    won = all(res["verdict"][str(c)]["qe_gains"] for c in cals)
    meas = res["dprime_empirical"]["mean_ratio"]
    pred = res["registered_prediction"]["predicted_dprime_ratio"]
    verdict_txt = (
        "VERDICT: qe GAINS on cars — D3 beats the folklore gate" if won else
        f"VERDICT: qe HARMS — measured $d'$-ratio {meas:.2f} vs (I)-model {pred}:\n"
        "selection effects (V1/V2) are first-order at mid-$h$; "
        "D3's margin→size link held ($d'$ fell ⇒ sets grew)")
    fig.suptitle(
        f"stanford_cars (K=196, $h_w$={res['registered_prediction']['h_w']}, "
        f"$h^*$={res['registered_prediction']['h_star']}) — {verdict_txt}",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    path = os.path.join(out_dir, "cars_qe_gate_verdict.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def fig_dprime(res, out_dir):
    dp = res["dprime_empirical"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    # reconstruct histogram support from summary + per-pair ratios if present
    ratios = np.array(dp.get("ratios", []))
    if ratios.size:
        ax.hist(ratios, bins=40, color="#4c72b0", alpha=0.75,
                label="per-pair empirical ratio")
    ax.axvline(1.0, color="k", lw=1.2, label="break-even (ratio = 1)")
    ax.axvline(res["registered_prediction"]["predicted_dprime_ratio"],
               color="tab:purple", ls="--", lw=2,
               label=f"D3 prediction "
                     f"{res['registered_prediction']['predicted_dprime_ratio']}")
    ax.axvline(dp["mean_ratio"], color="#dd8452", lw=2,
               label=f"measured mean {dp['mean_ratio']:.2f}")
    ax.set_xlabel("d'-ratio (smoothed / raw) on fixed nearest-prototype pair axes")
    ax.set_ylabel("number of class pairs")
    ax.set_title(
        f"stanford_cars — margin-level effect of qe, raw space\n"
        f"{dp['n_pairs']} pairs, {100 * dp['frac_pairs_improved']:.0f}% improved; "
        f"mean d': {dp['mean_raw_dprime']:.2f} → {dp['mean_smoothed_dprime']:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "cars_qe_gate_dprime.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def fig_pred_vs_measured(pv, out_dir):
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    lim = (0.4, 2.7)
    ax.plot(lim, lim, color="0.6", lw=1, ls="--", label="perfect (I)-model")
    ax.axhline(1.0, color="k", lw=1)
    ax.axvline(1.0, color="k", lw=1)
    ax.fill_between(lim, lim[0], 1.0, color="tab:red", alpha=0.06)
    color = {"gain": "tab:green", "harm": "tab:red"}
    label_pos = {"cifar100": (1.42, 1.28), "miniimagenet": (1.62, 1.52),
                 "cifar10": (2.18, 1.62), "stanford_cars": (1.60, 0.70),
                 "aircraft": (0.76, 0.72)}
    for ds, d in pv.items():
        x, ymeas = d["predicted_ratio"], d["measured_mean_ratio"]
        ax.scatter([x], [ymeas], s=80, color=color[d["verdict"]],
                   edgecolor="k", linewidth=0.6, zorder=3)
        ax.annotate(f"{ds}\n({ymeas:.2f} vs {x:.2f})", (x, ymeas),
                    xytext=label_pos[ds], fontsize=8,
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.7))
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("(I)-model predicted $d'$-ratio  (composition constants only)")
    ax.set_ylabel("measured $d'$-ratio  (selection effects included)")
    ax.set_title(
        "The idealization error, localized: every measured ratio sits BELOW\n"
        "the (I)-model prediction (V1/V2 always damp), and the measured sign\n"
        "predicts the CP verdict perfectly (green = qe gains, red = harms)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path = os.path.join(out_dir, "dprime_predicted_vs_measured.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="output/cars_qe_gate")
    args = ap.parse_args()
    with open(os.path.join(args.out_dir, "cars_qe_gate.json")) as f:
        res = json.load(f)
    fig_verdict(res, args.out_dir)
    fig_dprime(res, args.out_dir)
    pv_path = os.path.join(args.out_dir, "dprime_predicted_vs_measured.json")
    if os.path.exists(pv_path):
        with open(pv_path) as f:
            fig_pred_vs_measured(json.load(f), args.out_dir)


if __name__ == "__main__":
    main()
