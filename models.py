# imports
import pdb
from typing import DefaultDict
from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
import functions
from tqdm import tqdm 

# classes 
class CPH():
    def __init__(self, data):

        self.data = data
        self.params = DefaultDict()

    def set_random_hps(self):
        # weight decay or L2
        self.params["wd"] = np.power(10, np.random.uniform(-10,-9))
        # set number of input PCs
        self.params["nIN"] = np.random.randint(2, self.data.folds[0].train.x.shape[1])

    def _train(self, fold_index):
        # create lifelines dataset
        ds = pd.DataFrame(self.data.folds[fold_index].train.x.iloc[:,:self.params["nIN"]])
        ds["T"] = self.data.folds[fold_index].train.y["t"]
        ds["E"] = self.data.folds[fold_index].y["e"]
        CPH = CoxPHFitter(penalizer = self.params["wd"], l1_ratio = 0.)
        self.model = CPH.fit(ds, duration_col = "T", event_col = "E")
        l = self.model.log_likelihood_
        c = self.model.concordance_index_
    
    def _test(self, fold_index):
        self.out = self.model.predict_log_partial_hazard(self.data.folds[fold_index].test.x)
        return self.out 

class CPHDNN():
    def __init__(self, data):
        self.data = data

def train_test(data, model_type, input):
    # define data
    data.set_input_targets(input)
    data.shuffle()
    data.split_train_test(0.2)
    if model_type == "CPH":
        model = CPH(data)
    elif model_type == "CPHDNN":
        model = CPHDNN(data)
    else: model = None
    model._train()
    out = model._test()
    c_index = functions.compute_c_index(data.test.y["t"], data.test.y["e"], out)
    print (f"C index for model {model_type}, input: {input}: {c_index}")
    pdb.set_trace()

model_picker = {"CPH": CPH, "CPHDNN": CPHDNN}
def hpoptim(data, model_type, n = 100, nfolds = 5):
    # choose correct model, init
    model = model_picker[model_type](data)
    # split train / test (5)
    model.data.split_train_test(nfolds = 5) # 5 fold cross-val
    res = []
    # for each replicate (100)
    for rep_n in range(n):
        # fix (choose at random) set of HPs
        model.set_random_hps()    

        c_index = []
        # cycle through folds
        for fold_n in tqdm(range (nfolds), desc = f"N{rep_n + 1} Internal Cross Val"):
            # train
            model._train(fold_n)
            # test 
            out = model._test(fold_n)
            # record accuracy
            c_index.append(functions.compute_c_index(model.data.folds[fold_n].test.y["t"], model.data.folds[fold_n].test.y["e"], out))
        # compute aggregated c_index
        print(model.params, round(np.mean(c_index), 3))
        res.append(np.concatenate([[model.params[key] for key in ["wd", "nIN"]], [round(np.mean(c_index ),3)]] ))

        # for each fold (5)
            # train epochs (400)
            # test
        # record agg score, hps
    res = pd.DataFrame(res, columns = ["wd", "nIN", "c_index"] )
    res = res.sort_values(["c_index"], ascending = False)
    
    # return model
    return res

def main():
    # some test funcs
    pass

if __name__ == "__main__":
    main()