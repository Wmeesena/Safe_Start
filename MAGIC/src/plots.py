import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

def _ma(v, win):
    if not win or win <= 1: 
        return np.asarray(v, dtype=float)
    v = np.asarray(v, dtype=float)
    kernel = np.ones(win, dtype=float) / float(win)
    return np.convolve(v, kernel, mode="valid")

def plot_loss_multi(histories, which="total", smooth=None, title=None, gamma=None, savepath=None, show_log = False):
    """
    histories: list of (label, history_dict)
    which: "total" | "avg" | "rare" | "rare_weighted"
    smooth: moving average window (e.g., 5) or None
    gamma: if which == 'rare_weighted', multiply rare term by this gamma before plotting
    savepath: if given, save the figure to this path
    """
    key_map = {
        "total": "loss",
        "avg":   "loss_avg",
        "rare":  "loss_rare",
        "rare_weighted": "loss_rare"
    }
    assert which in key_map, f"which must be one of {list(key_map.keys())}"
    k = key_map[which]

    plt.figure()
    for label, h in histories:
        x = np.asarray(h["epoch"])
        y = np.asarray(h[k], dtype=float)
        if which == "rare_weighted":
            if gamma is None:
                raise ValueError("Provide gamma when which='rare_weighted'")
            y = gamma * y

        y_s = _ma(y, smooth)
        x_s = x if not smooth or smooth <= 1 else x[(smooth-1):]

        plt.plot(x_s, y_s+0.0000001, label=label)
    if show_log:
        plt.ylabel("log loss")
        plt.yscale("log")
    else:
        plt.ylabel("loss")
 

    ttl = title or f"Training {which} loss"
    plt.title(ttl)
    plt.xlabel("epoch")
    plt.legend(ncols=2 if len(histories) > 6 else 1)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

def plot_all_loss(results, show_opts=("adam","sgd"), smooth=5, gamma_for_weighted=None, save_prefix=None, show_log = False):
    """
    results: dict { (cfg,opt): (model, history) }
    show_opts: tuple/list of opts to include (e.g., ("adam",) to hide sgd)
    smooth: moving-average window for all plots
    gamma_for_weighted: if not None, also plot γ * rare
    save_prefix: if provided, save figs as f"{save_prefix}_{kind}.png"
    """
    # Build label->history list in stable order
    order_cfg = ["naive", "safe", "safe_neg"]
    order_opt = ["adam", "sgd"]
    hist_list = []
    # for cfg in order_cfg:
    #     for opt in order_opt:
    #         if opt not in show_opts: 
    #             continue
    #         if (cfg, opt) in results:
    #             _, hist = results[(cfg, opt)]
    #             label = f"{cfg}-{opt}"
    #             hist_list.append((label, hist))

    hist_list = []
    for cfg, opt in results.keys():
        if opt not in show_opts:

            continue
        _, hist = results[(cfg, opt)]
        label = f"{cfg}-{opt}"
        hist_list.append((label, hist))

        

    # Plots
    plot_loss_multi(hist_list, which="total", smooth=smooth, 
                    title="Total loss (all runs)",
                    savepath=(f"{save_prefix}_total.png" if save_prefix else None), show_log = show_log)
    plot_loss_multi(hist_list, which="avg", smooth=smooth, 
                    title="Average BCE loss (all runs)",
                    savepath=(f"{save_prefix}_avg.png" if save_prefix else None), show_log = show_log)
    plot_loss_multi(hist_list, which="rare", smooth=smooth, 
                    title="Rare-event loss (all runs)",
                    savepath=(f"{save_prefix}_rare.png" if save_prefix else None), show_log = show_log)
    if gamma_for_weighted is not None:
        plot_loss_multi(hist_list, which="rare_weighted", smooth=smooth, gamma=gamma_for_weighted,
                        title=f"γ-weighted rare-event loss (γ={gamma_for_weighted})",
                        savepath=(f"{save_prefix}_rare_weighted.png" if save_prefix else None), show_log = show_log)



def results_to_df(
    final_results: dict,
    *,
    # pretty labels for the config key (e.g., 'naive', 'safe', etc.)
    config_map: dict | None = None,
    # per-config safe bias values, e.g. {'safe': 15, 'safe_neg': 15}
    safe_bias_map: dict | None = None,
    # global run metadata (use None if not applicable)
    sigma: float | None = None,
    gamma: float | None = None,
    num_epochs: int | None = None,
    # display options
    percent: bool = True,           # convert metrics to %
    ci_decimals: int = 2,           # decimals for CI strings
    val_decimals: int = 2,          # decimals for value columns
):
    """
    Convert nested results to a tidy DataFrame with CI columns.
    final_results structure (example):
      {
        step_size: {
          (config_key, opt): {
             'acc': float, 'acc_ci': (lo, hi),
             'RA': float,  'RA_ci': (lo, hi),
             'CRA': float, 'CRA_ci': (lo, hi)
          }, ...
        }, ...
      }
    """
    config_map = config_map or {}
    safe_bias_map = safe_bias_map or {}

    def _p(x):
        return 100.0 * x if (x is not None and percent) else x

    def _fmt(val, lo_hi):
        if val is None or lo_hi is None:
            return ""
        lo, hi = lo_hi
        val, lo, hi = _p(val), _p(lo), _p(hi)
        return f"{val:.{val_decimals}f} [{lo:.{ci_decimals}f}, {hi:.{ci_decimals}f}]"

    rows = []
    for step_size, runs in final_results.items():
        for (cfg_key, opt), m in runs.items():
            # raw numbers
            acc, acc_ci = m.get('acc'), m.get('acc_ci')
            ra,  ra_ci  = m.get('RA'),  m.get('RA_ci')
            cra, cra_ci = m.get('CRA'), m.get('CRA_ci')

            # formatted strings
            acc_ci_str = _fmt(acc, acc_ci)
            ra_ci_str  = _fmt(ra,  ra_ci)
            cra_ci_str = _fmt(cra, cra_ci)

            # human labels
            opt_lbl = str(opt).upper()
            cfg_lbl = config_map.get(cfg_key, cfg_key)

            rows.append({
                "Opt": opt_lbl,
                "Config": cfg_lbl,
                "Safe Bias": safe_bias_map.get(cfg_key, "-"),
                "Step Size": step_size,
                "Sigma": sigma if sigma is not None else "-",
                "Gamma": gamma if gamma is not None else "-",
                "Number of Epochs": num_epochs if num_epochs is not None else "-",

                # numeric cols (already % if percent=True)
                "Test Acc": None if acc is None else round(_p(acc), val_decimals),
                "Test Acc (95% CI)": acc_ci_str,
                "RA": None if ra is None else round(_p(ra), val_decimals),
                "RA (95% CI)": ra_ci_str,
                "CRA": None if cra is None else round(_p(cra), val_decimals),
                "CRA (95% CI)": cra_ci_str,
            })

    df = pd.DataFrame(rows)

    # sort for readability
    sort_cols = ["Opt", "Step Size", "Config"]
    for c in sort_cols:
        if c not in df.columns:
            sort_cols.remove(c)
    if sort_cols:
        df = df.sort_values(sort_cols, ascending= [True, False, True]).reset_index(drop=True)

    return df