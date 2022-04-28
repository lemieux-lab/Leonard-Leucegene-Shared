from collections import defaultdict
import torch
from torch import nn 
import numpy as np

def _ridge_CPH(train_data):
    params = defaultdict()
    params["nepochs"] = 200
    params["nfolds"] = 5
    params["input_size"] = train_data.folds[0].train.x.shape[1] # dataset dependent!
    # weight decay or L2
    params["wd"] = np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
    params["W"] = 143 # np.random.randint(3,2048) # V2 Reasonable
    params["D"] = 2 # np.random.randint(2,4) # V2 Reasonable
    # self.params["dp"] = np.random.uniform(0,0.5) # cap at 0.5 ! (else nans in output)
    params["nL"] = np.random.choice([nn.ReLU()]) 
    params["ARCH"] = {
    "W": np.concatenate( [[  ## ARCHITECTURE ###
    params["input_size"]], ### INSIZE
        np.ones(params["D"] - 1) * params["W"], ### N hidden = D - 1  
        [1]]).astype(int), ### OUTNODE 
        "nL": np.array([params["nL"] for i in range(params["D"])]),
        # "dp": np.ones(self.params["D"]) * self.params["dp"] 
        }
    params["nInPCA"] = np.random.randint(2,26)
    params["device"] = "cuda:0"
    params["lr"] = 1e-4
    return params

def generate_default(model_type, data):
    picker = {
        "ridge_cph": _ridge_CPH
    }

    return picker[model_type](data)

