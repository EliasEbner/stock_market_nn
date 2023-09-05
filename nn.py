from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.l1(x)
        x = nn.functional.relu(x)
        x = self.l2(x)
        x = nn.functional.relu(x)
        x = self.l3(x)
        x = nn.functional.relu(x)
        x = self.l4(x)
        x = nn.functional.relu(x)
        x = self.l5(x)
        return x
