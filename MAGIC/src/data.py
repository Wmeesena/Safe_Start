from ucimlrepo import fetch_ucirepo 
import torch



def load_magic_data():
    '''
    load magic gamma telescope data from ucimlrepo and format them in torch tensors

    Returns:
        X_train: torch tensor of shape (11954, 10)
        y_train: torch tensor of shape (11954,)
        X_test: torch tensor of shape (4020, 10)
        y_test: torch tensor of shape (11954,)
    
    '''
    magic_gamma_telescope = fetch_ucirepo(id=159) 
    
    # data (as pandas dataframes) 
    X_df = magic_gamma_telescope.data.features 
    y_df = magic_gamma_telescope.data.targets 

    y_df = (y_df == 'g').astype(float)

    X_train_df = X_df.sample(n=15000, random_state=41)
    X_test_df = X_df.drop(X_train_df.index)

    y_train_df = y_df.loc[X_train_df.index]
    y_test_df = y_df.drop(y_train_df.index)


    X_train = torch.tensor(X_train_df.values, dtype=torch.float32)
    X_test = torch.tensor(X_test_df.values, dtype=torch.float32)

    y_train = torch.tensor(y_train_df.values, dtype=torch.float32)
    y_test = torch.tensor(y_test_df.values, dtype=torch.float32)



    return X_train, y_train, X_test, y_test
