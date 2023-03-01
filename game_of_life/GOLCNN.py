import numpy as np
import torch
from torch import nn

class OPNet(nn.Module):
    '''
    - m refers to the factor of overcompleteness of parameters in the CNN
    - n  refers to the number of steps of the Game of Life this CNN simulates.
    This is consistent with the convention used in the Kenyon and Springer paper
    about learning the Game of Life.
    '''
    def __init__(self, m=2, n=1):
        super(OPNet, self).__init__()
        
        self.overcompleteness_factor = m
        self.num_steps = n
        
        self.type1_layers = nn.ModuleList()
        self.type2_layers = nn.ModuleList()
        
        for i in range(self.num_steps):
            type1_layer = nn.Conv2d(m, m*2, 3, padding=1, padding_mode='circular')
            self.type1_layers.append(type1_layer)
        
            type2_layer = nn.Conv2d(m*2, m, 1)
            self.type2_layers.append(type2_layer)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
        self.final_conv_layer = nn.Conv2d(m, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        
    def forward(self, x):
        out=x
        
        for i in range(self.num_steps):
            out = self.type1_layers[i](out)
            out = self.relu1(out)

            out = self.type2_layers[i](out)
            out = self.relu2(out)
        
        out = self.final_conv_layer(out)
        out = self.sigmoid3(out)
        return out
    
    '''
    This function is meant for inspecting the intermediary results of the
    model. That is, it produces the output after n pairs of intermediary
    convolutional layers have been passed which corresponds to how the
    model "simulates" n steps of the game of Life.
    '''
    def inspect_step(self, x, n):
        
        if(n > self.num_steps):
            print(f'Error: {n} is more steps than the model has layers for.')
        else:
            out=x
            
            for i in range(n):
                out = self.type1_layers[i](out)
                out = self.relu1(out)

                out = self.type2_layers[i](out)
                out = self.relu2(out)
                
            out = self.final_conv_layer(out)
            out = self.sigmoid3(out)
            return out
        
        
def train_epoch(model, opt, criterion, train_loader, filters, device="cuda"):
    model.train()
    losses = []
    
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.repeat(1,filters,1,1)
        x_batch = x_batch.to(device)
        
        y_batch = y_batch.to(device)
        y_batch = y_batch[:,None,:,:]
        
        opt.zero_grad()
        y_hat = model(x_batch)
        loss = criterion(y_hat, y_batch)
        loss.backward()
        opt.step()
        losses.append(loss.cpu().data.numpy())
    
    return np.mean(losses)

def test_model(model, test_loader, filters, criterion, device="cuda"):
    model.eval()
    size = len(test_loader.dataset)
    losses = []
    num_correct = 0
    
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.repeat(1,filters,1,1)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        batch_outputs = model(x_batch)[:,0,:,:]
        batch_loss = criterion(batch_outputs, y_batch).cpu().data.numpy()
        losses.append(batch_loss)
        
        batch_preds = batch_outputs.cpu().detach().numpy()
        batch_preds = batch_preds.round()
        y_batch = y_batch.cpu().detach().numpy()
        
        for (pred, y) in zip(batch_preds, y_batch):
            num_correct += np.all(np.equal(batch_preds, y_batch))
    
    avg_loss = np.mean(losses)
    acc = num_correct / size
    num_wrong = size - num_correct
    
    return acc, avg_loss, num_correct, num_wrong