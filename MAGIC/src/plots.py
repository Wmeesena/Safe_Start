import numpy as np
import matplotlib.pyplot as plt
import torch

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
