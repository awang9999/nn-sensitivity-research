import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from scipy.stats import mode

def train_epoch(model, opt, criterion, train_loader, device="cuda", batch_size=64):
    model.train()
    losses = []
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        opt.zero_grad()
        y_hat = model(x_batch)
        loss = criterion(y_hat.flatten(), y_batch)
        loss.backward()
        opt.step()
        losses.append(loss.cpu().data.numpy())
    return np.mean(losses)

def eval_model(model, test_loader, device="cuda"):
    model.eval()
    preds = []
    labels = []
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        
        batch_outputs = model(x_batch)
        batch_preds = batch_outputs.cpu().detach().numpy()
        batch_labels = y_batch.numpy()
        preds.append(batch_preds)
        labels.append(batch_labels)
        
    preds = np.concatenate(preds).round().flatten()
    labels = np.concatenate(labels)
    
    diff = (preds == labels)
    acc = diff.sum() / diff.size
    return acc, preds, labels

# Assumes every model is equally weighted and uses majority vote to decide final label
def eval_ensemble(list_of_models, test_loader, device="cuda"):
    all_preds = []
    all_labels = []
    
    for model in list_of_models:
        _, preds, labels = eval_model(model, test_loader, device)
        all_preds.append(preds)
        all_labels.append(labels)
        
    all_preds = np.vstack(all_preds).T
    all_labels = all_labels[0]
    
    pred_modes, counts = mode(all_preds, axis=1, keepdims=False)
    
    diff = (pred_modes == labels)
    acc = diff.sum() / diff.size
        
    return acc, pred_modes, labels