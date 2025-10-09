import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader

# --- optional: small speed win on CUDA for fixed-size batches ---
torch.backends.cudnn.benchmark = True

from contextlib import nullcontext

def pretrain(X, y, model, num_epochs=10000, lr=1e-3, batch_size=256, use_amp=True, device=None):
    """
    Pretrain with BCEWithLogitsLoss on mini-batches, GPU if available.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)

    ds = TensorDataset(X, y)  # X,y should be on CPU for pin_memory to work best
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    amp_enabled = use_amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if amp_enabled else nullcontext()

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs", dynamic_ncols=True):
        model.train()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).squeeze(-1).float()

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                outputs = model(xb).squeeze(-1)
                loss = criterion(outputs, yb)

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

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
    sgd_momentum=0.0, sgd_nesterov=False,
    use_amp=True, device=None
):
    """
    Train with average BCE + γ * rare-event margin, mini-batches, GPU if available.
    Shows a tqdm bar over epochs (and an inner bar over batches).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    torch.manual_seed(42)

    # init model from pretrained
    new_model = type(pre_model)(X.shape[-1]).to(device)
    new_model.load_state_dict(pre_model.state_dict(), strict=False)

    if IF_SAFE:
        with torch.no_grad():
            new_model.fc_last.bias.fill_(SAFE_BIAS)

    # Keep data on CPU; move per batch
    ds = TensorDataset(X, y)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True
    )

    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    if callable(opt):
        optimizer = opt(new_model.parameters())
    else:
        key = str(opt).lower()
        if key == "adam":
            optimizer = optim.Adam(new_model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        elif key == "sgd":
            optimizer = optim.SGD(new_model.parameters(), lr=lr/10.0, momentum=sgd_momentum,
                                  nesterov=sgd_nesterov, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown opt='{opt}'")

    # AMP (new API)
    amp_enabled = use_amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if amp_enabled else nullcontext()

    history = {"epoch": [], "loss": [], "loss_avg": [], "loss_rare": [], "optimizer": str(opt)}

    # Epoch progress bar
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs", dynamic_ncols=True):
        new_model.train()
        tot = tot_avg = tot_rare = 0.0
        n_batches = 0

        # Inner per-batch bar (set leave=False to keep output clean)
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).squeeze(-1).float()

            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx:
                logits_clean = new_model(xb).squeeze(-1)
                loss_avg = criterion(logits_clean, yb)
                loss_rare = _rare_event_margin_binary(new_model, xb, yb, sigma=sigma, J=num_samples)
                loss = loss_avg + gamma * loss_rare

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            tot += float(loss)
            tot_avg += float(loss_avg)
            tot_rare += float(loss_rare)
            n_batches += 1

        epoch_loss = tot / max(1, n_batches)
        epoch_avg  = tot_avg / max(1, n_batches)
        epoch_rare = tot_rare / max(1, n_batches)

        history["epoch"].append(epoch)
        history["loss"].append(epoch_loss)
        history["loss_avg"].append(epoch_avg)
        history["loss_rare"].append(epoch_rare)

        # Show running stats on the epoch bar
        if epoch % 10 == 0: 
            tqdm.write(f"[epoch {epoch:03d}] loss={epoch_loss:.4f} | avg={epoch_avg:.4f} | rare={epoch_rare:.4f}")

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