
import torch
import torch.nn as nn



# Define the MLP model
class MLP(nn.Module):
    '''
    class for a simple Multi-Layer Perceptron (MLP) for binary classification.
    '''

    def __init__(self, input_size, num_layers = 2):
        super().__init__()
        self.fc_first = nn.Linear(input_size, 20)  # First hidden layer

        self.hidden_layers = nn.ModuleList()

        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(20, 20))
    
        self.fc_last = nn.Linear(20, 1)            # Output layer for binary classification
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc_first(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.fc_last(x)
        return x
 