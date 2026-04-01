import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import time


args = {
    'batch_size': 20,
    'num_workers': 4,
    'lr': 1e-4,
    'weight_decay': 5e-4,
    'num_epochs': 30,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

tfs = transforms.ToTensor()
train_set = datasets.MNIST('./', train=True, transform=tfs, download=True)
test_set = datasets.MNIST('./', train=False, transform=tfs, download=False)

train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'])
test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'])

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(MLP, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.out = nn.Linear(hidden_size, out_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = X.view(X.size(0), -1) # Flatten da imagem 28x28 para 784
        feature = self.features(X)
        return self.softmax(self.out(feature))

net = MLP(28*28, 128, 10).to(args['device'])

criterion = nn.CrossEntropyLoss().to(args['device'])
optimizer = optim.Adam(net.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

for epoch in range(args['num_epochs']):
    start = time.time()
    epoch_loss = []
    
    for batch in train_loader:
        dado, rotulo = batch
        dado, rotulo = dado.to(args['device']), rotulo.to(args['device'])
        
        optimizer.zero_grad()
        pred = net(dado)
        loss = criterion(pred, rotulo)
        loss.backward()
        optimizer.step()
        
        epoch_loss.append(loss.cpu().data.item())

    print(f"Epoca {epoch}, Loss: {np.mean(epoch_loss):.4f}, Tempo: {time.time()-start:.2f}s")
