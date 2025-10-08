import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_loss_multi(histories, which="total", smooth=None, title=None):
    """
    histories: list of (label, history_dict)
    which: "total" | "avg" | "rare"
    smooth: moving average window (e.g., 5) or None
    """
    key_map = {"total": "loss", "avg": "loss_avg", "rare": "loss_rare"}
    k = key_map[which]

    # def ma(v, win):
    #     if not win or win <= 1: return v
    #     kernel = np.ones(win) / win
    #     return np.convolve(v, kernel, mode="valid")

    plt.figure()
    for label, h in histories:

        x = np.array(h["epoch"])
        y = np.array(h[k])
        plt.plot(x, y, label=label)

    ttl = title or f"Training {which} loss"
    plt.title(ttl)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_all_loss(results):
    '''
    plot all losses from multiple runs
    Inputs:
    results: array of ( ((cfg, opt), hist) )

    '''
    # build label->history list
    hist_list_total = []
    for (cfg, opt), (model, hist) in results.items():
        label = f"{cfg}-{opt}"
        if opt != "sgd": hist_list_total.append((label, hist)) 

    # Compare TOTAL loss (recommended first look)
    plot_loss_multi(hist_list_total, which="total", smooth=5, title="Total loss (all runs)")

    # If you also want to compare the rare-event term only:
    plot_loss_multi(hist_list_total, which="rare", smooth=5, title="Rare-event loss (all runs)")

    # And the average BCE term:
    plot_loss_multi(hist_list_total, which="avg", smooth=5, title="Average BCE loss (all runs)")
