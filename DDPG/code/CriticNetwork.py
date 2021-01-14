import torch

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400

class CriticNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(CriticNetwork, self).__init__()
        self.layers = torch.nn.Sequential(
                    # torch.nn.BatchNorm1d(state_size + action_size),
                    torch.nn.Linear(state_size + action_size, HIDDEN1_UNITS),
                    torch.nn.ReLU(),
                    # torch.nn.BatchNorm1d(HIDDEN1_UNITS),
                    torch.nn.Linear(HIDDEN1_UNITS, HIDDEN2_UNITS),
                    torch.nn.ReLU(),
                    # torch.nn.BatchNorm1d(HIDDEN2_UNITS),
                    torch.nn.Linear(HIDDEN2_UNITS, action_size)
            )

    def forward(self, s, a):
        inp = torch.cat((s, a), axis=1)
        return self.layers(inp)
