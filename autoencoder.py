import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
# from training_data import x_train2,y_train,x_test2,y_test
from test2 import x_train,y_train
from testing import x_test,y_test

# Define the VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        '''
        self.encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.mean = nn.Linear(16, 8)
        self.logvar = nn.Linear(16, 8)

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        '''
            
        self.encoder = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU()

        )

        self.mean = nn.Linear(25, 2)
        self.logvar = nn.Linear(25, 2)

        self.decoder = nn.Sequential(
            nn.Linear(2, 25),
            nn.ReLU(),
            nn.Linear(25,50),
            nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100,10)
        )

    def encode(self, x):
        x = self.encoder(x)
        mean = self.mean(x)
        logvar = self.logvar(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

# Define the VAE loss function
def vae_loss(x_hat, x, mean, logvar):
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# Define the training function
def train(model, dataloader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0.0

        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            optimizer.zero_grad()

            x_hat, mean, logvar = model(inputs)
            loss = vae_loss(x_hat, inputs, mean, logvar)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        if((epoch+1)%100 == 0):
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {average_loss:.4f}")

# Prepare the data
#eigenvalues = torch.Tensor([[1.2, 0.8], [0.5, 0.3], [0.9, 1.5], [1.8, 1.2]])
#labels = torch.LongTensor([1, 0, 1, 0])

dataset = data.TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True)

# Set up the model, optimizer, and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the VAE
num_epochs = 2000
train(model, dataloader, optimizer, num_epochs)


# --------------------------------TESTING-----------------------------------------
# Define the testing function
def test(model, dataloader):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    latent_space_values = []	
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            x_hat, mean, logvar = model(inputs)
            loss = vae_loss(x_hat, inputs, mean, logvar)
            total_loss += loss.item()

            z = model.reparameterize(mean, logvar)
            latent_space_values.append(z)

    average_loss = total_loss / len(dataloader)
    print(f"Testing Loss: {average_loss:.4f}")


    latent_space_values = torch.cat(latent_space_values, dim=0)
    print("Latent Space Values:")
    print(latent_space_values)

# Prepare the testing data
test_dataset = data.TensorDataset(torch.Tensor(x_test), torch.LongTensor(y_test))
test_dataloader = data.DataLoader(test_dataset, batch_size=2, shuffle=False)

# Test the trained VAE
test(model, test_dataloader)
