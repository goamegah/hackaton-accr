import torch

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(MLP, self).__init__()

        # Couches de l'architecture MLP
        self.fc1 = torch.nn.Linear(input_dim, 8)
        self.fc2 = torch.nn.Linear(8, 16)
        self.fc3 = torch.nn.Linear(16, 32)
        self.fc4 = torch.nn.Linear(32, output_dim)
        
        # Couches de normalisation par lot
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        
        # Dropout pour régularisation
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)  
        x = self.fc4(x) 
        return x

    
