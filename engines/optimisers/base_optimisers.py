from engines.models.cox_models import CPH, CPHDNN, CoxSGD
import engines.models.functions as functions 
from torch import nn
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import pandas as pd
import pdb

model_picker = {"CPH": CPH, "CPHDNN": CPHDNN, "CoxSGD": CoxSGD}
class HPOptimiser:
    def __init__(self, train_data, 
                    model_type = "CPH", 
                    n = 100, 
                    nfolds = 5, 
                    nepochs = 1,
                    pca_input_size_range = [9,10]) -> None:
        self.params = None # the parameters!
        self.data = train_data
        self.model_type = model_type    
        self.model = model_picker[model_type]
        self.nepochs = nepochs
        self.pca_input_size_range = pca_input_size_range
        self.nfolds = nfolds
        self.n = n

    def run(self):
    
        # choose correct model, init
        model = self.model(self.data)
        # split train / test (5)
        model.data.split_train_test(nfolds = 5, device = ["cpu","cuda:0"][self.model_type == "CPHDNN"]) # 5 fold cross-val
        res = []
        best_params = None
        best_c_index = 0
        hpsearch = HPSearch(self.model_type, self.n, nepochs = self.nepochs)
        models = []
        # for each replicate (100)
        for rep_n, params in enumerate(hpsearch.params_list):
            
            # fix (choose at random) set of params
            model.set_fixed_params(params)
            tr_c_index = []
            scores = []
            print(rep_n, params)
            
            # cycle through folds
            for fold_n in tqdm(range (self.nfolds), desc = f"{self.model_type} - N{rep_n + 1} - Internal Cross Val"):
  
                # train
                tr_c = model._train_cv(fold_n)
                # test 
                out,l,c = model._valid_cv(fold_n)
                scores.append(out)
                # record accuracy
                tr_c_index.append(tr_c)
            c_ind_agg = functions.compute_c_index(model.data.y["t"], model.data.y["e"], np.concatenate(scores))
            c_ind_tr = np.mean(tr_c_index)
            if c_ind_agg > best_c_index:
                best_c_index = c_ind_agg
                best_params = model.params
            res.append([rep_n, c_ind_tr, c_ind_agg])
        res = pd.DataFrame(res, columns = ["repn", "c_ind_tr", "c_index_vld"])
        res = res.sort_values(["c_index_vld"], ascending = False)
        # RERUN model with best HPs
        opt_model = model_picker[self.model_type](self.data)
        opt_model.set_fixed_params(best_params)
        opt_model._train()
        # return model
        return res, opt_model

class HPSearch:
    def __init__(self, model_type, nrep, nepochs = 100) -> None:
        self.nrep = nrep
        self.model_type = model_type
        self.lr = 1e-3 # fixed 
        self.nepochs = nepochs
        self.params_list_generators = {
            "CPH": self.set_params_list_cph,
            "CoxSGD": self.set_params_list_coxsgd, 
            "CPHDNN": self.set_params_list_cphdnn
        }
        self.params_list = self.params_list_generators[self.model_type]()
        
    def set_params_list_cphdnn(self):
        params_list = []
        for i in range(self.nrep):
            params = defaultdict()
            params["wd"] = np.power(10, np.random.uniform(-10, -8)) # sample WD  
            params["lr"] = self.lr
            params["epochs"] = self.nepochs 
            params["D"] = np.random.randint(2,4)
            params["W"] = np.random.randint(2,2048)
            params["nL"] = nn.ReLU()
            params["opt_nepochs"] = 300 # fixed
            params["device"] = "cuda:0"
            params_list.append(params)
        return params_list 

    def set_params_list_coxsgd(self):
        params_list = []
        for i in range(self.nrep):
            params = defaultdict()
            params["wd"] = np.power(10, np.random.uniform(-10, -9)) # sample WD  
            params["lr"] = self.lr
            params["epochs"] = self.nepochs 
            params["opt_nepochs"] = 500 # fixed
            params_list.append(params)
        return params_list 

    def set_params_list_cph(self, input_size_range = 17):
        params_list = []
        for i in range(self.nrep):
            params = defaultdict()
            params["wd"] = np.power(10, np.random.uniform(-10, -9)) # sample WD  
            params["input_size"] = input_size_range
            params_list.append(params)
        return params_list 
    