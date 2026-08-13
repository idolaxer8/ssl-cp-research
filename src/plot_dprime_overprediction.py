"""Figure for the d'-overprediction investigation.

Reads output/cars_qe_gate/dprime_overprediction.json and renders one
three-panel figure localizing WHERE the (I)-model's error lives:

  panel 1  signal shrink S: predicted vs measured per dataset
  panel 2  noise shrink N: predicted (rho) vs measured per dataset
  panel 3  the denominator decomposition -- measured Var(nu_v)/sigma_v^2 and
           2(1-beta)beta-weighted Cov(e_v, nu_v)/sigma_v^2 vs the (I)-model
           values (1/k_eff and 0)

Linear axes.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["cifar10", "miniimagenet", "cifar100", "stanford_cars", "aircraft"]
SHORT = {"cifar10": "c10", "miniimagenet": "mini", "cifar100": "c100",
         "stanford_cars": "cars", "aircraft": "airc"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="output/cars_qe_gate")
    args = ap.parse_args()
    with open(os.path.join(args.out_dir, "dprime_overprediction.json")) as f:
        res = json.load(f)
    ds_list = [d for d in ORDER if d in res]
    x = np.arange(len(ds_list))
    wid = 0.38

    fig, (axS, axN, axD) = plt.subplots(1, 3, figsize=(14, 4.4))

    for ax, key_p, key_m, title, model_lab in [
            (axS, "S_pred", "S_meas",
             "Signal shrink $S$ (mean-separation multiplier)\n"
             "model uses composition constants only", "(I)-model $S$"),
            (axN, "N_pred", "N_meas",
             "Noise shrink $N$ (within-class sd multiplier)\n"
             "model: $\\rho=\\sqrt{(1-\\beta)^2+\\beta^2/k_{\\rm eff}}$",
             "(I)-model $\\rho$")]:
        pred = [res[d][key_p] for d in ds_list]
        meas = [res[d][key_m] for d in ds_list]
        ax.bar(x - wid / 2, pred, wid, color="0.75", label=model_lab)
        ax.bar(x + wid / 2, meas, wid, color="#4c72b0", label="measured")
        ax.axhline(1.0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[d] for d in ds_list])
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    axS.set_ylabel("multiplier (1 = unchanged)")

    # panel 3: denominator variance components (units of sigma_v^2)
    vnu_m = [res[d]["var_nu_over_sigma2"] for d in ds_list]
    vnu_p = [res[d]["var_nu_model"] for d in ds_list]
    cov_m = [res[d]["cov_e_nu_over_sigma2"] for d in ds_list]
    axD.bar(x - wid / 2, vnu_p, wid, color="0.75",
            label="model $\\mathrm{Var}(\\nu_v)/\\sigma_v^2 = 1/k_{\\rm eff}$")
    axD.bar(x + wid / 2, vnu_m, wid, color="#dd8452",
            label="measured $\\mathrm{Var}(\\nu_v)/\\sigma_v^2$")
    axD.bar(x + wid / 2, cov_m, wid, bottom=vnu_m, color="#c44e52",
            label="measured $\\mathrm{Cov}(e_v,\\nu_v)/\\sigma_v^2$ (model: 0)")
    axD.set_xticks(x)
    axD.set_xticklabels([SHORT[d] for d in ds_list])
    axD.set_title("Where the noise floor comes from: the neighbor\n"
                  "mean carries the ego's own local structure", fontsize=10)
    axD.set_ylabel("variance components  (units of $\\sigma_v^2$)")
    axD.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Decomposing the (I)-model's universal d'-ratio overprediction: "
        "ratio = S / N — which factor is wrong, and why", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(args.out_dir, "dprime_overprediction.png")
    fig.savefig(path, dpi=150)
    print("saved", path)


if __name__ == "__main__":
    main()
