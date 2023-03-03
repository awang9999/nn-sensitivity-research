import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from GameOfLife import StandardEngine

def generateDataset(dataSetSize=1000, size=32, n_steps=2, returnTensor=False):
    init_state = np.zeros((32,32))
    engine = StandardEngine(init_state)
    
    inputs = []
    outputs = []
    
    for i in range(dataSetSize):
        start_state = np.random.randint(0,2,(size,size))
        inputs.append(start_state)
        
        engine.game_state = start_state
        engine.step_n(n_steps)
        
        output_state = engine.game_state
        outputs.append(output_state)
    
    inputs = np.array(inputs)
    inputs = inputs.reshape((dataSetSize, 1, size, size))
    outputs = np.array(outputs)
    ouputs = outputs.reshape((dataSetSize, 1, size, size))
    
    tensor_x = torch.Tensor(inputs)
    tensor_y = torch.Tensor(outputs)
    
    dataSet = TensorDataset(tensor_x, tensor_y)
    if returnTensor:
        return dataSet
    
    dataLoader = DataLoader(dataSet)
    return dataLoader