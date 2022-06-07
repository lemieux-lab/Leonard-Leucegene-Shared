from collections import defaultdict
import torch
from torch import nn 
import numpy as np
from datetime import datetime
import hashlib 

class HP_dict:
    def __init__(self, wd, nepochs, bootstr_n, nfolds) -> None:
        self.WD = wd
        self.nepochs = nepochs
        self.bootstr_n = bootstr_n
        self.nfolds = nfolds

    def _ridge_cph_lifelines(self, train_data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = self.nepochs
        params["nfolds"] = self.nfolds
        params["cohort"] = train_data.name
        params["input_size"] = train_data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 1 # np.random.randint(3,2048) # V2 Reasonable
        # self.params["dp"] = np.random.uniform(0,0.5) # cap at 0.5 ! (else nans in output)
        params["nL"] = np.random.choice([nn.ReLU()]) 
        params["ARCH"] = {"W": [params["input_size"],params["W"]], "nl": None}
        params["input_type"] = None
        params["device"] = "cuda:0"
        params["lr"] = 1e-4
        params["linear"] = True
        params["modeltype"] = "ridge_cph_lifelines"
        params["bootstrap_n"] = self.bootstr_n
        return params

    def _CPHDNN_1l(self, train_data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = self.nepochs
        params["nfolds"] = self.nfolds
        params["cohort"] = train_data.name
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
        params["input_type"] = None
        params["device"] = "cuda:0"
        params["lr"] = 1e-4
        params["linear"]  = False
        params["modeltype"] = "CPHDNN"
        params["bootstrap_n"] = self.bootstr_n
        return params
    
    def _CPHDNN_2l(self, train_data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = self.nepochs
        params["nfolds"] = self.nfolds
        params["cohort"] = train_data.name
        params["input_size"] = train_data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 143 # np.random.randint(3,2048) # V2 Reasonable
        params["D"] = 3 # np.random.randint(2,4) # V2 Reasonable
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
        params["input_type"] = None
        params["device"] = "cuda:0"
        params["lr"] = 1e-4
        params["linear"]  = False
        params["modeltype"] = "CPHDNN"
        params["bootstrap_n"] = self.bootstr_n
        return params
    
    def _CPHDNN_3l(self, train_data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = self.nepochs
        params["nfolds"] = self.nfolds
        params["cohort"] = train_data.name
        params["input_size"] = train_data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 143 # np.random.randint(3,2048) # V2 Reasonable
        params["D"] = 4 # np.random.randint(2,4) # V2 Reasonable
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
        params["input_type"] = None
        params["device"] = "cuda:0"
        params["lr"] = 1e-4
        params["linear"]  = False
        params["modeltype"] = "CPHDNN"
        params["bootstrap_n"] = self.bootstr_n
        return params
    
    def _CPHDNN_4l(self, train_data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = self.nepochs
        params["nfolds"] = self.nfolds
        params["cohort"] = train_data.name
        params["input_size"] = train_data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 143 # np.random.randint(3,2048) # V2 Reasonable
        params["D"] = 5 # np.random.randint(2,4) # V2 Reasonable
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
        params["input_type"] = None
        params["device"] = "cuda:0"
        params["lr"] = 1e-4
        params["linear"]  = False
        params["modeltype"] = "CPHDNN"
        params["bootstrap_n"] = self.bootstr_n
        return params
    
    def _CPHDNN_5l(self, train_data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = self.nepochs
        params["nfolds"] = self.nfolds
        params["cohort"] = train_data.name
        params["input_size"] = train_data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 143 # np.random.randint(3,2048) # V2 Reasonable
        params["D"] = 6 # np.random.randint(2,4) # V2 Reasonable
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
        params["input_type"] = None
        params["device"] = "cuda:0"
        params["lr"] = 1e-4
        params["linear"]  = False
        params["modeltype"] = "CPHDNN"
        params["bootstrap_n"] = self.bootstr_n
        return params
    
    def _CPHDNN_6l(self, train_data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = self.nepochs
        params["nfolds"] = self.nfolds
        params["cohort"] = train_data.name
        params["input_size"] = train_data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 143 # np.random.randint(3,2048) # V2 Reasonable
        params["D"] = 7 # np.random.randint(2,4) # V2 Reasonable
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
        params["input_type"] = None
        params["device"] = "cuda:0"
        params["lr"] = 1e-4
        params["linear"]  = False
        params["modeltype"] = "CPHDNN"
        params["bootstrap_n"] = self.bootstr_n
        return params

    def _CPHDNN_7l(self, train_data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = self.nepochs
        params["nfolds"] = self.nfolds
        params["cohort"] = train_data.name
        params["input_size"] = train_data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 143 # np.random.randint(3,2048) # V2 Reasonable
        params["D"] = 8 # np.random.randint(2,4) # V2 Reasonable
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
        params["input_type"] = None
        params["device"] = "cuda:0"
        params["lr"] = 1e-4
        params["linear"]  = False
        params["modeltype"] = "CPHDNN"
        params["bootstrap_n"] = self.bootstr_n
        return params

    def _CPHDNN_8l(self, train_data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = self.nepochs
        params["nfolds"] = self.nfolds
        params["cohort"] = train_data.name
        params["input_size"] = train_data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 143 # np.random.randint(3,2048) # V2 Reasonable
        params["D"] = 9 # np.random.randint(2,4) # V2 Reasonable
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
        params["input_type"] = None
        params["device"] = "cuda:0"
        params["lr"] = 1e-4
        params["linear"]  = False
        params["modeltype"] = "CPHDNN"
        params["bootstrap_n"] = self.bootstr_n
        return params
    
    def _CPHDNN_9l(self, train_data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = self.nepochs
        params["nfolds"] = self.nfolds
        params["cohort"] = train_data.name
        params["input_size"] = train_data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 143 # np.random.randint(3,2048) # V2 Reasonable
        params["D"] = 10 # np.random.randint(2,4) # V2 Reasonable
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
        params["input_type"] = None
        params["device"] = "cuda:0"
        params["lr"] = 1e-4
        params["linear"]  = False
        params["modeltype"] = "CPHDNN"
        params["bootstrap_n"] = self.bootstr_n
        return params

    def _CPHDNN_10l(self, train_data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = self.nepochs
        params["nfolds"] = self.nfolds
        params["cohort"] = train_data.name
        params["input_size"] = train_data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 143 # np.random.randint(3,2048) # V2 Reasonable
        params["D"] = 11 # np.random.randint(2,4) # V2 Reasonable
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
        params["input_type"] = None
        params["device"] = "cuda:0"
        params["lr"] = 1e-4
        params["linear"]  = False
        params["modeltype"] = "CPHDNN"
        params["bootstrap_n"] = self.bootstr_n
        return params


    def _cyto_risk(self, data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = 0
        params["nfolds"] = 0
        params["cohort"] = data.name
        params["input_size"] = 0 # dataset dependent!
        # weight decay or L2
        params["wd"] = 0 #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 0 # np.random.randint(3,2048) # V2 Reasonable
        params["D"] = 0 # np.random.randint(2,4) # V2 Reasonable
        # self.params["dp"] = np.random.uniform(0,0.5) # cap at 0.5 ! (else nans in output)
        params["nL"] = 0
        params["ARCH"] = None
        params["input_type"] = "cytogenetic risk"
        params["device"] = None
        params["lr"] = None
        params["linear"]  = False
        params["modeltype"] = "cytogenetic risk"
        params["bootstrap_n"] = self.bootstr_n
        return params 

    def _ridge_cph_lifelines_lsc17(self, data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = 0
        params["nfolds"] = self.nfolds
        params["cohort"] = data.name
        params["input_size"] = data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 0 # np.random.randint(3,2048) # V2 Reasonable
        params["D"] = 0 # np.random.randint(2,4) # V2 Reasonable
        # self.params["dp"] = np.random.uniform(0,0.5) # cap at 0.5 ! (else nans in output)
        params["nL"] = 0
        params["ARCH"] = None
        params["input_type"] = "LSC17"
        params["device"] = None
        params["lr"] = None
        params["linear"]  = True
        params["modeltype"] = "ridge_cph_lifelines"
        params["bootstrap_n"] = self.bootstr_n
        return params
    def _ridge_cph_lifelines_PCA(self, data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = 0
        params["nfolds"] = self.nfolds
        params["cohort"] = data.name
        params["input_size"] = data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 0 # np.random.randint(3,2048) # V2 Reasonable
        params["D"] = 0 # np.random.randint(2,4) # V2 Reasonable
        # self.params["dp"] = np.random.uniform(0,0.5) # cap at 0.5 ! (else nans in output)
        params["nL"] = 0
        params["ARCH"] = None
        params["input_type"] = "PCA"
        params["device"] = None
        params["lr"] = None
        params["linear"]  = True
        params["modeltype"] = "ridge_cph_lifelines"
        params["bootstrap_n"] = self.bootstr_n
        return params
    
    def _ridge_cph_lifelines_CF(self, data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = 0
        params["nfolds"] = self.nfolds
        params["cohort"] = data.name
        params["input_size"] = data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 0 # np.random.randint(3,2048) # V2 Reasonable
        params["D"] = 0 # np.random.randint(2,4) # V2 Reasonable
        # self.params["dp"] = np.random.uniform(0,0.5) # cap at 0.5 ! (else nans in output)
        params["nL"] = 0
        params["ARCH"] = None
        params["input_type"] = "clinical factors"
        params["device"] = None
        params["lr"] = None
        params["linear"]  = True
        params["modeltype"] = "ridge_cph_lifelines"
        params["bootstrap_n"] = self.bootstr_n
        return params
    def _ridge_cph_lifelines_CF_LSC17(self, data):
        params = defaultdict()
        params["model_id"] = hashlib.sha1(str(datetime.now()).encode()).hexdigest() # create random id for storage purposes 
        params["nepochs"] = 0
        params["nfolds"] = self.nfolds
        params["cohort"] = data.name
        params["input_size"] = data.folds[0].train.x.shape[1] # dataset dependent!
        # weight decay or L2
        params["wd"] = self.WD #np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        params["W"] = 0 # np.random.randint(3,2048) # V2 Reasonable
        params["D"] = 0 # np.random.randint(2,4) # V2 Reasonable
        # self.params["dp"] = np.random.uniform(0,0.5) # cap at 0.5 ! (else nans in output)
        params["nL"] = 0
        params["ARCH"] = None
        params["input_type"] = "clinical factors + LSC17 expressions"
        params["device"] = None
        params["lr"] = None
        params["linear"]  = True
        params["modeltype"] = "ridge_cph_lifelines"
        params["bootstrap_n"] = self.bootstr_n
        return params

    

    def generate_default(self, model_type, data):
        picker = {
            "ridge_cph_lifelines_PCA": self._ridge_cph_lifelines_PCA,
            "ridge_cph_lifelines_LSC17": self._ridge_cph_lifelines_lsc17,
            "ridge_cph_lifelines_CDS": self._ridge_cph_lifelines,
            "ridge_cph_lifelines_CF": self._ridge_cph_lifelines_CF,
            "ridge_cph_lifelines_CF_LSC17": self._ridge_cph_lifelines_CF_LSC17,
            "cphdnn_1l": self._CPHDNN_1l,
            "cphdnn_2l": self._CPHDNN_2l,
            "cphdnn_3l": self._CPHDNN_3l,
            "cphdnn_4l": self._CPHDNN_4l,
            "cphdnn_5l": self._CPHDNN_5l,
            "cphdnn_6l": self._CPHDNN_6l,
            "cphdnn_7l": self._CPHDNN_7l,
            "cphdnn_8l": self._CPHDNN_8l,
            "cphdnn_9l": self._CPHDNN_9l,
            "cphdnn_10l": self._CPHDNN_10l,
            "cytogenetic_risk": self._cyto_risk
        }

        return picker[model_type](data)

