import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np



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



def evaluate_avg_accuracy(X, y, model):
    '''
    evaluate the average accuracy of the model on clean input

    Parameters:
    - X: Input features (tensor).
    - y: Target labels (tensor).
    - model: The neural network model to be evaluated.
    Returns:
    - accuracy: The average accuracy of the model. (float)
    
    '''
    with torch.no_grad():
        
        outputs = model(X).squeeze()
        predicted = (torch.sigmoid(outputs) >= 0.5).float()
        
        accuracy = (predicted == y.squeeze().float()).float().mean().item()
        
        print(f'Accuracy: {accuracy * 100:.2f}%')

    return accuracy




def evaluate_robust(X, y, model, num_samples = 100, sigma = 0.1):
    '''
    evaluate the robust accuracy of the model on Gaussian noise corrupted input

    Parameters:
    - X: Input features (tensor). (num_data, num_features)
    - y: Target labels (tensor). (num_data, 1)
    - model: The neural network model to be evaluated.
    - num_samples: Number of noisy samples to generate for each input.
    
    Returns:
    - RE: Robust error of the model. (float)
    - CRE: conditional robust error of the model for the correctly predicted input
    '''

    with torch.no_grad():
    
        torch.manual_seed(42)  # Set the seed for reproducibility
        X_noise = X + sigma * torch.randn(num_samples, *X.shape)
    
        y_noise = y.expand(num_samples, * y.shape)


        outputs = model(X).squeeze(-1)
        predicted = (torch.sigmoid(outputs) >= 0.5).float()
        correct = (predicted ==  y.squeeze(-1)).float()

        outputs_noise = model(X_noise).squeeze(-1)
        predicted_noise = (torch.sigmoid(outputs_noise) >= 0.5).float()
        accuracy = (predicted_noise ==  y_noise.squeeze(-1).float()).float().mean(axis = 0).flatten()
        RA = accuracy.mean().item()
        CRA = (accuracy[correct == 1].mean().item() if correct.sum() > 0 else 0)


        print(f'Robust Accuracy: {RA * 100:.5f}%')
        print(f'Conditional Robust Accuracy: {CRA * 100:.5f}%')
        
        # print(f'Accuracy: {accuracy * 100:.10f}%')

    return RA, CRA






# def evaluate_robust(X, y, model, num_samples = 100, sigma = 0.1, quantile = 0.001):
#     '''
#     evaluate the robust accuracy of the model on Gaussian noise corrupted input

#     Parameters:
#     - X: Input features (tensor).
#     - y: Target labels (tensor).
#     - model: The neural network model to be evaluated.
#     - num_samples: Number of noisy samples to generate for each input.
#     - sigma: Standard deviation of the Gaussian noise.
#     Returns:
#     - accuracy: The robust accuracy of the model per input. (tensor)
#     '''

#     with torch.no_grad():
    
#         torch.manual_seed(42)  # Set the seed for reproducibility
#         X_noise = X + sigma * torch.randn(num_samples, *X.shape)
    
#         y_noise = y.expand(num_samples, * y.shape)

#         outputs = model(X_noise).squeeze(-1)
#         predicted = (torch.sigmoid(outputs) >= 0.5).float()
#         accuracy = (predicted ==  y_noise.squeeze(-1).float()).float().mean(axis = 0).flatten()
#         mean_accuracy = accuracy.mean().item()
#         var = accuracy.kthvalue(int(quantile * accuracy.numel())+1).values.item()
        
#         print(f'Mean Accuracy: {mean_accuracy * 100:.5f}%')
#         print(f'{quantile} Quantile Accuracy: {var * 100:.5f}%')
        
#         # print(f'Accuracy: {accuracy * 100:.10f}%')

#     return mean_accuracy, var




# def joint_train(X, y, X_robust, y_robust, pretrained_model, num_epochs = 100, gamma = 1000, num_samples = 10, sigma = 0.1, IF_SAFE = False):
#     '''
#     train a model using both clean and Gaussian noise corrupted data (Lagrangian formulation)
#     Parameters:
#     - X: Input features (Tensor).
#     - y: Target labels (Tensor).
#     - X_robust: Robust input features (Tensor).
#     - y_robust: Robust target labels (Tensor).
#     - pretrained_model: The pretrained neural network model to be further trained.
#     - num_epochs: Number of training epochs.
#     - gamma: Weight for the robust loss component.
#     - num_samples: Number of noisy samples to generate for each robust input.
#     - sigma: Standard deviation of the Gaussian noise.
#     - IF_SAFE: If True, modify the last layer's bias to a large value to ensure initial robust accuracy.
#     Returns:
#     - model: The trained model.

    
#     '''
#     torch.manual_seed(42)
#     X_robust_noise = X_robust + sigma * torch.randn(num_samples, *X_robust.shape)
#     y_robust_noise = y_robust.expand(num_samples, * y_robust.shape)
    
 
#     new_model = type(pretrained_model)( X.shape[-1])  # Create a new instance of the model
#     # new_model.load_state_dict(pretrained_model.state_dict())  # Copy the weights

#     to_load = dict(pretrained_model.named_parameters())
#     new_model.load_state_dict(to_load, strict=False) 
#     if IF_SAFE:
#         with torch.no_grad():
#             new_model.fc_last.bias.fill_(1000)  # Assuming the last layer is named 'fc' and setting bias to 1000

#      # Define the loss function and optimizer
#     criterion = nn.BCEWithLogitsLoss()  # Suitable for binary classification
#     optimizer = optim.Adam(new_model.parameters(), lr=0.001)


#     # Training loop
#     for epoch in range(num_epochs):
#         # Copy the pretrained model parameters to a new model
        
#         new_model.train()  # Set the model to training mode
        
#         # Zero the gradients
#         optimizer.zero_grad()
        
#         # Forward pass
#         outputs = new_model(X).squeeze(-1)  # Get model predictions and remove extra dimensions
#         outputs_robust = new_model(X_robust_noise).squeeze(-1)
#         loss = criterion(outputs, 
#                         y.squeeze(-1)) + gamma *criterion(outputs_robust, 
#                                                          y_robust_noise.squeeze(-1))# Compute the loss
        
#         # Backward pass and optimization
#         loss.backward()
#         optimizer.step()
        
#         # if (epoch+1) % 10 == 0 or epoch == 0:
#         #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
#     return new_model





def safe_train(X, y, X_robust, y_robust, pretrianed_model, num_epochs = 100):
    return 0


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

def joint_train(X, y, X_robust, y_robust, pretrained_model, num_epochs = 100,
                gamma = 1000, num_samples = 10, sigma = 0.1, IF_SAFE = False, SAFE_BIAS = 1000):
    '''
    train a model with average BCE + γ * rare-event margin (only on clean-correct points).
    '''
    torch.manual_seed(42)

    # --- (unchanged) init model from pretrained ---
    new_model = type(pretrained_model)(X.shape[-1])
    to_load = dict(pretrained_model.named_parameters())
    new_model.load_state_dict(to_load, strict=False)

    if IF_SAFE:
        with torch.no_grad():
            new_model.fc_last.bias.fill_(SAFE_BIAS)  # if your last layer is named fc_last

    # --- losses & optimizer (unchanged average loss) ---
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(new_model.parameters(), lr=0.001)

    # --- training loop ---
    for epoch in range(num_epochs):
        print(epoch)
        new_model.train()
        optimizer.zero_grad()

        # Average-case loss (clean)
        logits_clean = new_model(X).squeeze(-1)                 # (B,)
        loss_avg = criterion(logits_clean, y.squeeze(-1).float())

        # Rare-event loss on noisy inputs, masked by clean-correct indicator
        loss_rare = _rare_event_margin_binary(
            new_model, X, y, sigma=sigma, J=num_samples
        )

        loss = loss_avg + gamma * loss_rare
        loss.backward()
        optimizer.step()

    return new_model