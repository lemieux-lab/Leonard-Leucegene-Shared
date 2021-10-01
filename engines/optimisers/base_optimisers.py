from engines.models.cox_models import CPH, CPHDNN
import engines.models.functions as functions 
from tqdm import tqdm
import numpy as np
import pandas as pd
import pdb
model_picker = {"CPH": CPH, "CPHDNN": CPHDNN}
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
        model = self.model(self.data, nepochs = self.nepochs)
        # split train / test (5)
        model.data.split_train_test(nfolds = 5) # 5 fold cross-val
        if self.model_type == "CPHDNN": model.data.folds_to_cuda_tensors()
        res = []
        best_params = None
        best_c_index = 0
        rep_params_list = functions.set_params_list(self.n, self.pca_input_size_range)
        models = []
        # for each replicate (100)
        for rep_n, params in enumerate(rep_params_list):
            
            # fix (choose at random) set of params
            model.set_fixed_params(params)
            tr_c_index = []
            scores = []
            # cycle through folds
            for fold_n in tqdm(range (self.nfolds), desc = f"{self.model_type} - N{rep_n + 1} - Internal Cross Val"):
  
                # train
                tr_c = model._train_cv(fold_n)
                # test 
                out,l,c = model._valid_cv(fold_n)
                scores.append(out)
                # record accuracy
                tr_c_index.append(tr_c)
            c_ind_agg = functions.compute_aggregated_c_index(scores, model.data)
            c_ind_tr = np.mean(tr_c_index)
            if c_ind_agg > best_c_index:
                best_c_index = c_ind_agg
                best_params = model.params
            # compute aggregated c_index
            # print(model.params, round(score, 3))
            if self.model_type == "CPHDNN":
                res.append(np.concatenate([[model.params[key] for key in ["wd", "input_size", "D","W"]], [c_ind_tr, c_ind_agg]] ))
            elif self.model_type == "CPH":
                res.append(np.concatenate([[model.params[key] for key in ["wd", "input_size"]], [c_ind_tr, c_ind_agg]] )) 
            # for each fold (5)
                # train epochs (400)
                # test
            # record agg score, params
            
        if self.model_type == "CPHDNN":
            res = pd.DataFrame(res, columns = ["wd", "nIN", "D", "W", "c_index_train", "c_index_vld"] )
        elif self.model_type == "CPH":
            res = pd.DataFrame(res, columns = ["wd", "nIN", "c_index_train", "c_index_vld"] ) 
        res = res.sort_values(["c_index_vld"], ascending = False)
        # RERUN model with best HPs
        opt_model = model_picker[self.model_type](self.data)
        opt_model.set_fixed_params(best_params)
        opt_model._train()
        # return model
        return res, opt_model