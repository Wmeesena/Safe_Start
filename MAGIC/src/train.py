import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- optional: small speed win on CUDA for fixed-size batches ---
torch.backends.cudnn.benchmark = True

def pretrain(X, y, model, num_epochs=10000, lr=1e-3, batch_size=256, use_amp=True, device=None):
    """
    Pretrain with BCEWithLogitsLoss on mini-batches, GPU if available.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    # keep tensors on CPU and move per-batch (scales to larger data),
    # but will also work if X/y are already on device.
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type=="cuda"))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type=="cuda"))

    for _ in range(num_epochs):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type=="cuda")):
                outputs = model(xb).squeeze(-1)
                loss = criterion(outputs, yb.squeeze(-1).float())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    return model


@torch.no_grad()
def _clean_correct_mask_binary(model, x, y):
    logits = model(x).squeeze(-1)
    pred = (torch.sigmoid(logits) >= 0.5).float()
    y_flat = y.squeeze(-1).float()
    return (pred == y_flat).float()

def _rare_event_margin_binary(model, x, y, sigma=0.1, J=10):
    """
    Vectorized rare-event margin; runs on whatever device x is on.
    """
    device = x.device
    B = x.shape[0]

    I_clean = _clean_correct_mask_binary(model, x, y).to(device)  # (B,)

    eps = sigma * torch.randn((J,) + x.shape, device=device)      # (J,B,...)
    x_noisy = x.unsqueeze(0) + eps
    x_flat  = x_noisy.reshape(J * B, *x.shape[1:])

    logits = model(x_flat).squeeze(-1)                            # (J*B,)
    y_rep  = y.squeeze(-1).float().unsqueeze(0).repeat(J,1).reshape(-1)

    s = (2.0 * y_rep - 1.0)
    g_y = s * logits
    margin = (-2.0 * g_y).clamp_min(0.0)

    margin_JB = margin.view(J, B)
    margin_mean = margin_JB.mean(dim=0)
    margin_masked = margin_mean * I_clean
    return margin_masked.mean()


def joint_train(
    X, y, pre_model, num_epochs=100,
    gamma=1000, num_samples=10, sigma=0.1, IF_SAFE=False, SAFE_BIAS=1000,
    batch_size=256,
    opt="adam", lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999),
    sgd_momentum=0.9, sgd_nesterov=False,
    use_amp=True, device=None
):
    """
    Train with average BCE + γ * rare-event margin, mini-batches, GPU if available.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(42)

    # init model from pretrained
    new_model = type(pre_model)(X.shape[-1]).to(device)
    new_model.load_state_dict(pre_model.state_dict(), strict=False)

    if IF_SAFE:
        with torch.no_grad():
            new_model.fc_last.bias.fill_(SAFE_BIAS)

    # Keep data on CPU; move per batch (works even if already on device)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=(device.type=="cuda"))

    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    if callable(opt):
        optimizer = opt(new_model.parameters())
    else:
        key = str(opt).lower()
        if key == "adam":
            optimizer = optim.Adam(new_model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        elif key == "sgd":
            optimizer = optim.SGD(new_model.parameters(), lr=lr, momentum=sgd_momentum,
                                  nesterov=sgd_nesterov, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown opt='{opt}'")

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type=="cuda"))

    history = {"epoch": [], "loss": [], "loss_avg": [], "loss_rare": [], "optimizer": str(opt)}

    for epoch in range(num_epochs):
        new_model.train()
        tot = tot_avg = tot_rare = 0.0
        n_batches = 0

        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type=="cuda")):
                logits_clean = new_model(xb).squeeze(-1)
                loss_avg = criterion(logits_clean, yb.squeeze(-1).float())
                loss_rare = _rare_event_margin_binary(new_model, xb, yb, sigma=sigma, J=num_samples)
                loss = loss_avg + gamma * loss_rare

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tot += loss.item(); tot_avg += loss_avg.item(); tot_rare += loss_rare.item()
            n_batches += 1

        history["epoch"].append(epoch + 1)
        history["loss"].append(tot / n_batches)
        history["loss_avg"].append(tot_avg / n_batches)
        history["loss_rare"].append(tot_rare / n_batches)

    return new_model, history


def train_all(X, y, pre_model, num_epochs, gamma, num_samples, sigma, device=None, **kw):
    """
    Train (naive, safe, safe_neg) × (Adam, SGD) on chosen device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}   # results[("naive","adam")] = (model, hist)

    for opt in ["adam", "sgd"]:
        print(f"\nTraining naive with {opt}...")
        naive_model, naive_hist = joint_train(
            X, y, pre_model, num_epochs=num_epochs, gamma=gamma,
            num_samples=num_samples, sigma=sigma, IF_SAFE=False,
            opt=opt, device=device, **kw
        )
        results[("naive", opt)] = (naive_model, naive_hist)

        print(f"\nTraining safe with {opt}...")
        safe_model, safe_hist = joint_train(
            X, y, pre_model, num_epochs=num_epochs, gamma=gamma,
            num_samples=num_samples, sigma=sigma, IF_SAFE=True, SAFE_BIAS=1000,
            opt=opt, device=device, **kw
        )
        results[("safe", opt)] = (safe_model, safe_hist)

        print(f"\nTraining safe_neg with {opt}...")
        safe_neg_model, safe_neg_hist = joint_train(
            X, y, pre_model, num_epochs=num_epochs, gamma=gamma,
            num_samples=num_samples, sigma=sigma, IF_SAFE=True, SAFE_BIAS=-1000,
            opt=opt, device=device, **kw
        )
        results[("safe_neg", opt)] = (safe_neg_model, safe_neg_hist)

    return results