import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import math


def pretrain(X, y, model, num_epochs=10000):
    '''
    Function to pretrain a given model using binary cross-entropy loss.
    
    Parameters:
    - X: Input features (tensor).
    - y: Target labels (tensor).
    - model: The neural network model to be trained.
    - num_epochs: Number of training epochs.
    - learning_rate: Learning rate for the optimizer.
    
    Returns:
    - model: The trained model.
    '''
    
    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Suitable for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X).squeeze()  # Get model predictions and remove extra dimensions
        loss = criterion(outputs, y.squeeze())  # Compute the loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # if (epoch+1) % 10 == 0 or epoch == 0:
        #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model


# --- helper: clean-correct indicator for binary logits (no grad) ---
@torch.no_grad()
def _clean_correct_mask_binary(model, x, y):
    """
    y: (B,1) or (B,) with {0,1}. Model returns logits (B,1) or (B,).
    Returns float mask (B,) in {0.,1.}
    """
    logits = model(x).squeeze(-1)              # (B,)
    pred = (torch.sigmoid(logits) >= 0.5).float()
    y_flat = y.squeeze(-1).float()
    return (pred == y_flat).float()            # (B,)

def _rare_event_margin_binary(model, x, y, sigma=0.1, J=10):
    """
    Implements E_ε[(max_{i≠y} g_i(x+ε;θ) - g_y(x+ε;θ))_+] * I{clean-correct}
    for binary with single logit z: let s = 2y-1 in {-1,+1}, g_y = s*z, g_other = -g_y,
    so margin = relu(g_other - g_y) = relu(-2*g_y) = relu(-2*s*z).
    """
    device = x.device
    B = x.shape[0]

    # Indicator I{ y = argmax_i g_i(x;θ) } on clean inputs (detached)
    I_clean = _clean_correct_mask_binary(model, x, y).to(device)  # (B,)

    # Sample J noises and forward once (vectorized)
    eps = sigma * torch.randn((J,) + x.shape, device=device)      # (J,B, ...)
    x_noisy = x.unsqueeze(0) + eps                                # (J,B,...)
    x_flat  = x_noisy.reshape(J * B, *x.shape[1:])                # (J*B, ...)

    logits = model(x_flat).squeeze(-1)                            # (J*B,)
    y_rep  = y.squeeze(-1).float().unsqueeze(0).repeat(J,1).reshape(-1)  # (J*B,)

    s = (2.0 * y_rep - 1.0)                                       # in {-1,+1}
    g_y = s * logits                                              # (J*B,)
    margin = (-2.0 * g_y).clamp_min(0.0)                          # relu(-2*g_y)

    # Average over J for each item, then apply indicator and batch-mean
    margin_JB = margin.view(J, B)
    margin_mean = margin_JB.mean(dim=0)                           # (B,)
    margin_masked = margin_mean * I_clean                          # (B,)
    return margin_masked.mean()                                    # scalar




def joint_train(
    X, y, pretrained_model, num_epochs=100,
    gamma=1000, num_samples=10, sigma=0.1, IF_SAFE=False, SAFE_BIAS=1000,
    batch_size=256,
    # --- new optimizer knobs ---
    opt="adam",             # "adam" or "sgd" or a callable(params)->optimizer
    lr=1e-3,
    weight_decay=0.0,       # works for both Adam/SGD
    betas=(0.9, 0.999),     # Adam only
    sgd_momentum=0.9,       # SGD only
    sgd_nesterov=False      # SGD only
):
    """
    Train with average BCE + γ * rare-event margin (only on clean-correct points),
    using mini-batches on CPU. Choose Adam or SGD via `opt`.
    """
    torch.manual_seed(42)

    # init model from pretrained (unchanged)
    new_model = type(pretrained_model)(X.shape[-1])
    to_load = dict(pretrained_model.named_parameters())
    new_model.load_state_dict(to_load, strict=False)

    if IF_SAFE:
        with torch.no_grad():
            new_model.fc_last.bias.fill_(SAFE_BIAS)

    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    criterion = nn.BCEWithLogitsLoss()

    # ----- optimizer factory -----
    if callable(opt):
        optimizer = opt(new_model.parameters())  # user-supplied factory
    else:
        key = str(opt).lower()
        if key == "adam":
            optimizer = optim.Adam(new_model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        elif key == "sgd":
            optimizer = optim.SGD(
                new_model.parameters(), lr=lr, momentum=sgd_momentum,
                nesterov=sgd_nesterov, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown opt='{opt}'. Use 'adam', 'sgd', or pass a callable.")

    # --- (optional) simple history for plotting if you added plot_loss earlier ---
    history = {"epoch": [], "loss": [], "loss_avg": [], "loss_rare": [], "optimizer": str(opt)}

    for epoch in range(num_epochs):
        new_model.train()
        tot = tot_avg = tot_rare = 0.0
        n_batches = 0

        for xb, yb in dl:
            optimizer.zero_grad()

            logits_clean = new_model(xb).squeeze(-1)
            loss_avg = criterion(logits_clean, yb.squeeze(-1).float())

            loss_rare = _rare_event_margin_binary(
                new_model, xb, yb, sigma=sigma, J=num_samples
            )

            loss = loss_avg + gamma * loss_rare
            loss.backward()
            optimizer.step()

            tot += loss.item(); tot_avg += loss_avg.item(); tot_rare += loss_rare.item()
            n_batches += 1

        history["epoch"].append(epoch + 1)
        history["loss"].append(tot / n_batches)
        history["loss_avg"].append(tot_avg / n_batches)
        history["loss_rare"].append(tot_rare / n_batches)

    return new_model, history




def train_all(X,y,pre_model,num_epochs, gamma, num_samples, sigma):
    '''
    train all models (naive, safe, safe_neg) with both Adam and SGD optimizers
    Inputs:
        X: input features (tensor)
        y: input labels (tensor)
        pre_model: pretrained model (nn.Module)
        num_epochs: number of epochs to train
        gamma: weight for rare-event loss
        num_samples: number of noise samples for rare-event loss
        sigma: noise level for rare-event loss
    Outputs:
        results: dictionary of trained models and their training histories
    '''

    results = {}   # e.g., results[("naive","adam")] = (model, hist)


    for opt in ["adam", "sgd"]:
        # naive
        print(f"\nTraining naive with {opt}...")
        naive_model, naive_hist = joint_train(
            X, y,
            pre_model,
            num_epochs=num_epochs, gamma=gamma, num_samples=num_samples, sigma=sigma ,
            IF_SAFE=False, opt=opt
        )
        results[("naive", opt)] = (naive_model, naive_hist)

        # safe (positive bias)
        print(f"\nTraining safe with {opt}...")
        safe_model, safe_hist = joint_train(
            X, y,
            pre_model,
            num_epochs=num_epochs, gamma=gamma, num_samples=num_samples, sigma=sigma,
            IF_SAFE=True, SAFE_BIAS=1000, opt=opt
        )
        results[("safe", opt)] = (safe_model, safe_hist)

        # safe_negative (negative bias)
        print(f"\nTraining safe_neg with {opt}...")
        safe_neg_model, safe_neg_hist = joint_train(
            X, y,
            pre_model,
            num_epochs=num_epochs, gamma=gamma, num_samples=num_samples, sigma=sigma,
            IF_SAFE=True, SAFE_BIAS=-1000, opt=opt
        )
        results[("safe_neg", opt)] = (safe_neg_model, safe_neg_hist)


    return results