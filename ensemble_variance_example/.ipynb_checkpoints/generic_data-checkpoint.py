import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

class GenericDataset(torch.utils.data.Dataset):
  '''
  Prepare dataset for regression
  '''

  def __init__(self, X, y, scale_data=True):
        
    # Apply scaling if necessary
    if scale_data:
        X = StandardScaler().fit_transform(X)
        
    self.X = torch.tensor(X,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.X.shape[0]

  def __len__(self):
      return self.length

  def __getitem__(self, i):
      return self.X[i], self.y[i]