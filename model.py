import torch
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        # Define your encoder network architecture (e.g., using nn.Linear layers)
        # ...
        #500-500-2000.
        self.fc1 = nn.Linear(input_dim, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 2000)
        self.mu = nn.Linear(2000, latent_dim)
        self.log_var = nn.Linear(2000, latent_dim)

    def forward(self, x):
        # Implement the forward pass of your encoder network
        # ...
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mu = self.mu(x)
        log_var = self.log_var(x)
        return mu, log_var

# --- Decoder Network ---
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        # Define your decoder network architecture (e.g., using nn.Linear layers)
        # ...
        #2000-500-500-784.
        self.fc1 = nn.Linear(latent_dim, 2000)
        self.fc2 = nn.Linear(2000, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, output_dim)

    def forward(self, z):
        # Implement the forward pass of your decoder network
        # ...
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        z = torch.relu(self.fc3(z))
        p = torch.sigmoid(self.fc4(z))
        return p
