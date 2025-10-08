from ucimlrepo import fetch_ucirepo 
import torch



def load_magic_data(device=None, train_n=15000, seed=41):
    """
    Load MAGIC Gamma Telescope and return torch tensors on `device`.
    device: "cuda", "cpu", or None (auto: cuda if available)
    Returns:
        X_train: (train_n, 10)
        y_train: (train_n,)
        X_test:  (~4020, 10)
        y_test:  (~4020,)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    magic_gamma_telescope = fetch_ucirepo(id=159) 
    
    # pandas DataFrames
    X_df = magic_gamma_telescope.data.features 
    y_df = magic_gamma_telescope.data.targets 

    # binary {0,1}
    y_df = (y_df == 'g').astype('float32')

    # train/test split by sampling
    X_train_df = X_df.sample(n=train_n, random_state=seed)
    X_test_df  = X_df.drop(X_train_df.index)

    y_train_df = y_df.loc[X_train_df.index]
    y_test_df  = y_df.loc[X_test_df.index]

    # -> torch tensors on device
    X_train = torch.tensor(X_train_df.values, dtype=torch.float32, device=device)
    X_test  = torch.tensor(X_test_df.values,  dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_df.values, dtype=torch.float32, device=device)
    y_test  = torch.tensor(y_test_df.values,  dtype=torch.float32, device=device)

    return X_train, y_train, X_test, y_test
