from collections import defaultdict
import torch
from torch import nn 
import numpy as np
from datetime import datetime
import hashlib 

class HP_dict:
    def __init__(self, args) -> None:
        self.WD = args.WEIGHT_DECAY
        self.nepochs = args.NEPOCHS
        self.bootstr_n = args.bootstr_n
        self.nfolds = args.NFOLDS
        self.cohort = args.COHORT

    def _ridge_cph_lifelines(self, train_data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = self.nepochs
        params["nfolds"] = self.nfolds
        params["cohort"] = self.cohort
        params["input_size"] = train_data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 1 # np.random.randint(3,2048) # V2 Reasonable
        # self.params["dp"] = np.random.uniform(0,0.5) # cap at 0.5 ! (else nans in output)
        params["nL"] = np.random.choice([nn.ReLU()]) 
        params["ARCH"] = {"W": [params["input_size"],params["W"]], "nl": None}
        params["pca_t"] = False
        params["device"] = "cuda:0"
        params["lr"] = 1e-4
        params["linear"] = True
        params["modeltype"] = "ridge_cph_lifelines"
        params["bootstrap_n"] = self.bootstr_n
        return params

    def _CPHDNN(self, train_data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = self.nepochs
        params["nfolds"] = self.nfolds
        params["cohort"] = self.cohort
        params["input_size"] = train_data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
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
        params["pca_t"] = False
        params["device"] = "cuda:0"
        params["lr"] = 1e-4
        params["linear"]  = False
        params["modeltype"] = "CPHDNN"
        params["bootstrap_n"] = self.bootstr_n
        return params

    def generate_default(self, model_type, data):
        picker = {
            "ridge_cph_lifelines": self._ridge_cph_lifelines,
            "ridge_cph": self._ridge_cph_lifelines,
            "cphdnn": self._CPHDNN
        }

        return picker[model_type](data)

