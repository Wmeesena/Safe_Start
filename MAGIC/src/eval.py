import torch
import math
import numpy as np
from src.utils import pct, pct_ci

# ---------- helpers ----------
def _wilson_interval(k, n, conf=0.95, eps=1e-12):
    """Wilson score CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)  # degenerate
    z = {0.80:1.2816, 0.90:1.6449, 0.95:1.96, 0.98:2.326, 0.99:2.576}.get(conf, 1.96)
    p = k / (n + eps)
    denom = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    halfwidth = (z * math.sqrt((p*(1-p))/n + (z*z)/(4*n*n))) / denom
    lo = max(0.0, center - halfwidth)
    hi = min(1.0, center + halfwidth)
    return (lo, hi)

def _bootstrap_mean_ci(values, conf=0.95, B=1000, seed=42):
    """Nonparametric bootstrap CI for the mean of a 1D array-like."""
    rng = np.random.default_rng(seed)
    vals = np.asarray(values, dtype=float)
    n = len(vals)
    if n == 0:
        return (0.0, 0.0)
    idx = rng.integers(0, n, size=(B, n))
    means = vals[idx].mean(axis=1)
    alpha = (1 - conf) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return (float(lo), float(hi))

# ---------- your functions with CI ----------
def evaluate_avg_accuracy(X, y, model, seed=42, conf=0.95):
    '''
    Returns:
      acc: float
      acc_ci: (lo, hi) Wilson 95% CI by default
    '''
    torch.manual_seed(seed)
    with torch.no_grad():
        outputs = model(X).squeeze(-1)
        predicted = (torch.sigmoid(outputs) >= 0.5).float()
        correct_vec = (predicted == y.squeeze(-1).float()).float()
        k = int(correct_vec.sum().item())
        n = int(correct_vec.numel())
        acc = k / max(n, 1)
        lo, hi = _wilson_interval(k, n, conf=conf)

        print(f'Accuracy: {acc*100:.2f}%  (CI {int(conf*100)}%: [{lo*100:.2f}%, {hi*100:.2f}%])')

    return acc, (lo, hi)

def evaluate_robust(X, y, model, num_samples=100, sigma=0.1, seed=42, conf=0.95, B=1000):
    '''
    Returns:
      RA: float, RA_CI: (lo, hi)
      CRA: float, CRA_CI: (lo, hi)   (0/degenerate if no clean-correct)
    '''
    torch.manual_seed(seed)
    with torch.no_grad():
        # noisy copies
        X_noise = X + sigma * torch.randn(num_samples, *X.shape)
        y_noise = y.expand(num_samples, *y.shape)

        # clean correctness mask
        outputs_clean = model(X).squeeze(-1)
        predicted_clean = (torch.sigmoid(outputs_clean) >= 0.5).float()
        correct_mask = (predicted_clean == y.squeeze(-1).float()).float()  # (N,)
        correct_bool = (correct_mask == 1)

        # noisy predictions -> per-example robust accuracy (mean over noises)
        outputs_noise = model(X_noise).squeeze(-1)                       # (J,N)
        predicted_noise = (torch.sigmoid(outputs_noise) >= 0.5).float()  # (J,N)
        acc_per_example = (predicted_noise == y_noise.squeeze(-1).float()).float().mean(dim=0)  # (N,)

        # RA: average across all examples
        RA = float(acc_per_example.mean().item())
        RA_lo, RA_hi = _bootstrap_mean_ci(acc_per_example.cpu().numpy(), conf=conf, B=B, seed=seed)

        # CRA: average across clean-correct examples only
        if correct_bool.any():
            acc_on_correct = acc_per_example[correct_bool]
            CRA = float(acc_on_correct.mean().item())
            CRA_lo, CRA_hi = _bootstrap_mean_ci(acc_on_correct.cpu().numpy(), conf=conf, B=B, seed=seed)
        else:
            CRA = 0.0
            CRA_lo, CRA_hi = 0.0, 0.0

        print(f'Robust Accuracy: {RA*100:.5f}%  (CI {int(conf*100)}%: [{RA_lo*100:.5f}%, {RA_hi*100:.5f}%])')
        print(f'Conditional Robust Accuracy: {CRA*100:.5f}%  (CI {int(conf*100)}%: [{CRA_lo*100:.5f}%, {CRA_hi*100:.5f}%])')

    return (RA, (RA_lo, RA_hi)), (CRA, (CRA_lo, CRA_hi))




def eval_one(model, X_test, y_test, sigma, SAMPLES_EVAL, conf=0.95, B=1000):
    acc, acc_ci = evaluate_avg_accuracy(X_test, y_test, model, conf=conf)
    (RA, RA_ci), (CRA, CRA_ci) = evaluate_robust(
        X_test, y_test, model, num_samples=2*SAMPLES_EVAL, sigma=sigma, conf=conf, B=B
    )
    return {
        "acc": acc, "acc_ci": acc_ci,
        "RA": RA, "RA_ci": RA_ci,
        "CRA": CRA, "CRA_ci": CRA_ci
    }


def eval_all(results, X_test, y_test, SIGMA, SAMPLES_EVAL, metrics):
    # Evaluate all
    for cfg in ["naive", "safe", "safe_neg"]:
        for opt in ["adam", "sgd"]:
            model, _hist = results[(cfg, opt)]
            metrics[(cfg, opt)] = eval_one(model, X_test, y_test, SIGMA, SAMPLES_EVAL)

    # Print compact summary
    print("\n=== Summary (mean ± 95% CI) ===")

    for cfg, opt in metrics:
        m = metrics[(cfg, opt)]
        line = (f"{cfg:9s} ({opt.upper():4s}):  "
                f"Acc {pct(m['acc'])} {pct_ci(m['acc_ci'])}  |  "
                f"RA {pct(m['RA'])} {pct_ci(m['RA_ci'])}  |  "
                f"CRA {pct(m['CRA'])} {pct_ci(m['CRA_ci'])}")
        print(line)




