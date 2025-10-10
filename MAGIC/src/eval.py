import torch
import math
import numpy as np
from src.utils import pct, pct_ci
from math import ceil

# ---------- helpers ----------
def _wilson_interval(k, n, conf=0.95, eps=1e-12):
    if n == 0: return (0.0, 0.0)
    z = {0.80:1.2816, 0.90:1.6449, 0.95:1.96, 0.98:2.326, 0.99:2.576}.get(conf, 1.96)
    p = k / (n + eps)
    denom = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    halfwidth = (z * math.sqrt((p*(1-p))/n + (z*z)/(4*n*n))) / denom
    lo = max(0.0, center - halfwidth)
    hi = min(1.0, center + halfwidth)
    return (lo, hi)

def _bootstrap_mean_ci(values, conf=0.95, B=1000, seed=42):
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n == 0: return (0.0, 0.0)
    idx = rng.integers(0, n, size=(B, n))
    means = vals[idx].mean(axis=1)
    alpha = (1 - conf) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return (float(lo), float(hi))

# ---------- device utility ----------
def _model_device(model):
    return next(model.parameters()).device

# ---------- eval with device awareness ----------
def evaluate_avg_accuracy(X, y, model, seed=42, conf=0.95):
    """
    Works on CPU or GPU; moves X,y to model's device if needed.
    """
    torch.manual_seed(seed)
    device = _model_device(model)
    with torch.no_grad():
        xb = X.to(device, non_blocking=True)
        yb = y.to(device, non_blocking=True)

        outputs = model(xb).squeeze(-1)
        predicted = (torch.sigmoid(outputs) >= 0.5).float()
        correct_vec = (predicted == yb.squeeze(-1).float()).float()
        k = int(correct_vec.sum().item())
        n = int(correct_vec.numel())
        acc = k / max(n, 1)
        lo, hi = _wilson_interval(k, n, conf=conf)

        print(f'Accuracy: {acc*100:.2f}%  (CI {int(conf*100)}%: [{lo*100:.2f}%, {hi*100:.2f}%])')
    return acc, (lo, hi)

# def evaluate_robust(X, y, model, num_samples=100, sigma=0.1, seed=42, conf=0.95, B=1000, J_chunk=None):
#     torch.manual_seed(seed)
#     device = _model_device(model)
#     dtype = X.dtype
#     with torch.no_grad():
#         xb = X.to(device, non_blocking=True)
#         yb = y.to(device, non_blocking=True)

#         # clean correctness mask (bool)
#         outputs_clean = model(xb).squeeze(-1)
#         predicted_clean = (torch.sigmoid(outputs_clean) >= 0.5)
#         correct_bool = (predicted_clean == yb.squeeze(-1).bool())

#         N = xb.shape[0]
#         J = int(num_samples)
#         if not J_chunk or J_chunk > J:
#             J_chunk = J

#         sum_correct = torch.zeros(N, device=device, dtype=torch.float32)
#         total_J = 0
#         remain = J
#         while remain > 0:
#             m = min(remain, J_chunk)
#             remain -= m
#             total_J += m

#             eps = sigma * torch.randn((m,) + xb.shape, device=device, dtype=dtype)  # (m,N,feat...)
#             x_noisy = xb.unsqueeze(0) + eps                                          # (m,N,feat...)
#             outputs_noise = model(x_noisy).squeeze(-1)                                # (m,N)
#             predicted_noise = (torch.sigmoid(outputs_noise) >= 0.5)
#             y_noise = yb.expand(m, *yb.shape)                                         # (m,N,1) or (m,N)
#             acc_chunk_sum = (predicted_noise == y_noise.squeeze(-1).bool()).float().sum(dim=0)  # (N,)
#             sum_correct += acc_chunk_sum

#         acc_per_example = sum_correct / max(total_J, 1)  # (N,)

#         # RA
#         RA = float(acc_per_example.mean().item())
#         RA_lo, RA_hi = _bootstrap_mean_ci(acc_per_example.detach().cpu().numpy(), conf=conf, B=B, seed=seed)

#         # CRA
#         if correct_bool.any():
#             acc_on_correct = acc_per_example[correct_bool]
#             CRA = float(acc_on_correct.mean().item())
#             CRA_lo, CRA_hi = _bootstrap_mean_ci(acc_on_correct.detach().cpu().numpy(), conf=conf, B=B, seed=seed)
#         else:
#             CRA, CRA_lo, CRA_hi = 0.0, 0.0, 0.0

#         print(f'Robust Accuracy: {RA*100:.5f}%  (CI {int(conf*100)}%: [{RA_lo*100:.5f}%, {RA_hi*100:.5f}%])')
#         print(f'Conditional Robust Accuracy: {CRA*100:.5f}%  (CI {int(conf*100)}%: [{CRA_lo*100:.5f}%, {CRA_hi*100:.5f}%])')

#     return (RA, (RA_lo, RA_hi)), (CRA, (CRA_lo, CRA_hi))


def eval_one(model, X_test, y_test, sigma, SAMPLES_EVAL, conf=0.95, B=1000, J_chunk=None):
    acc, acc_ci = evaluate_avg_accuracy(X_test, y_test, model, conf=conf)
    (RA, RA_ci), (CRA, CRA_ci) = evaluate_robust(
        X_test, y_test, model, num_samples=2*SAMPLES_EVAL, sigma=sigma, conf=conf, B=B, J_chunk=J_chunk
    )
    return {"acc": acc, "acc_ci": acc_ci, "RA": RA, "RA_ci": RA_ci, "CRA": CRA, "CRA_ci": CRA_ci}

def eval_all(results, X_test, y_test, SIGMA, SAMPLES_EVAL, order=None, **kw):
    # Evaluate 3×2
    metrics = {}  # (cfg,opt) -> metric dict
    for cfg, opt in results.keys():
        model, _hist = results[(cfg, opt)]
        metrics[(cfg, opt)] = eval_one(model, X_test, y_test, SIGMA, SAMPLES_EVAL, **kw)

    # Print compact summary (stable order)
    print("\n=== Summary (mean ± 95% CI) ===")
    for cfg, opt in results.keys():
        m = metrics[(cfg, opt)]
        line = (f"{cfg:9s} ({opt.upper():4s}):  "
                f"Acc {pct(m['acc'])} {pct_ci(m['acc_ci'])}  |  "
                f"RA {pct(m['RA'])} {pct_ci(m['RA_ci'])}  |  "
                f"CRA {pct(m['CRA'])} {pct_ci(m['CRA_ci'])}")
        print(line)

    return metrics


import torch
from math import ceil

from math import ceil
from contextlib import nullcontext

def evaluate_robust(
    X, y, model, num_samples=100, sigma=0.1, seed=42, conf=0.95, B=1000,
    J_chunk=None, N_chunk=None  # <-- only addition; optional
):
    """
    Same API as before; computes RA and CRA correctly.
    Minimal change: optional N_chunk to avoid (J,N,D) allocations.
    """
    torch.manual_seed(seed)
    device = _model_device(model)
    dtype = X.dtype
    model.eval()

    # Default: keep original behavior (no dataset chunking)
    if N_chunk is None or N_chunk <= 0:
        N_chunk = X.shape[0]

    # We'll collect per-example robust accuracies and clean-correct flags on CPU
    acc_per_example_parts = []
    clean_correct_parts = []

    with torch.no_grad():
        total_J = int(num_samples)  # for clarity

        # Iterate dataset in chunks to avoid holding (J, N, D) at once
        for start in range(0, X.shape[0], N_chunk):
            xb = X[start:start+N_chunk].to(device, non_blocking=True)
            yb = y[start:start+N_chunk].to(device, non_blocking=True)

            # ---- clean correctness (bool) per *batch* ----
            outputs_clean = model(xb).squeeze(-1)
            predicted_clean = (torch.sigmoid(outputs_clean) >= 0.5)
            correct_bool_batch = (predicted_clean == yb.squeeze(-1).bool())  # (B,)

            # ---- robust accuracy per example (batch) ----
            J = int(num_samples)
            if not J_chunk or J_chunk > J:
                J_chunk = J

            sum_correct = torch.zeros(xb.shape[0], device=device, dtype=torch.float32)
            remain = J
            while remain > 0:
                m = min(remain, J_chunk)
                remain -= m

                # only (m, B, D...), not (m, N, D...)
                eps = sigma * torch.randn((m,) + xb.shape, device=device, dtype=dtype)   # (m,B,D...)
                x_noisy = xb.unsqueeze(0) + eps                                          # (m,B,D...)
                outputs_noise = model(x_noisy).squeeze(-1)                                # (m,B)
                predicted_noise = (torch.sigmoid(outputs_noise) >= 0.5)
                y_noise = yb.expand(m, *yb.shape)                                         # (m,B,1) or (m,B)
                acc_chunk_sum = (predicted_noise == y_noise.squeeze(-1).bool()).float().sum(dim=0)  # (B,)
                sum_correct += acc_chunk_sum

                # free temporaries for safety (optional)
                del eps, x_noisy, outputs_noise, predicted_noise, y_noise

            acc_per_example_batch = (sum_correct / max(J, 1)).to('cpu')          # (B,)
            acc_per_example_parts.append(acc_per_example_batch)
            clean_correct_parts.append(correct_bool_batch.to('cpu'))

            # free batch tensors
            del xb, yb, outputs_clean, predicted_clean, correct_bool_batch, sum_correct

    # Stitch per-example vectors back together (CPU)
    acc_per_example = torch.cat(acc_per_example_parts)            # (N,)
    correct_bool = torch.cat(clean_correct_parts)                  # (N,) bool

    # ---- RA ----
    RA = float(acc_per_example.mean().item())
    RA_lo, RA_hi = _bootstrap_mean_ci(acc_per_example.numpy(), conf=conf, B=B, seed=seed)

    # ---- CRA (conditional on clean-correct) ----
    if correct_bool.any():
        acc_on_correct = acc_per_example[correct_bool]
        CRA = float(acc_on_correct.mean().item())
        CRA_lo, CRA_hi = _bootstrap_mean_ci(acc_on_correct.numpy(), conf=conf, B=B, seed=seed)
    else:
        CRA, CRA_lo, CRA_hi = 0.0, 0.0, 0.0

    print(f'Robust Accuracy: {RA*100:.5f}%  (CI {int(conf*100)}%: [{RA_lo*100:.5f}%, {RA_hi*100:.5f}%])')
    print(f'Conditional Robust Accuracy: {CRA*100:.5f}%  (CI {int(conf*100)}%: [{CRA_lo*100:.5f}%, {CRA_hi*100:.5f}%])')

    return (RA, (RA_lo, RA_hi)), (CRA, (CRA_lo, CRA_hi))
