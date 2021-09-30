from torch import nn 
from collections import defaultdict
import numpy as np
from lifelines import CoxPHFitter
import engines.models.functions as functions 
import pandas as pd
# classes 
class CPH():
    def __init__(self, data, nepochs = 1):

        self.data = data
        self.params = defaultdict()

    def set_random_params(self, input_size = None):
        # weight decay or L2
        self.params["wd"] = np.power(10, np.random.uniform(-10,-9))
        input_size = min(np.random.randint(2, self.data.folds[0].train.x.shape[1]), 50)
        # set number of input PCs
        self.params["input_size"] = input_size
    
    def set_fixed_params(self, params):
        self.params = params
    
    def _train(self):
        # create lifelines dataset
        ds = pd.DataFrame(self.data.x.iloc[:,:self.params["input_size"]])
        ds["T"] = self.data.y["t"]
        ds["E"] = self.data.y["e"]
        CPH = CoxPHFitter(penalizer = self.params["wd"], l1_ratio = 0.)
        self.model = CPH.fit(ds, duration_col = "T", event_col = "E")
        l = self.model.log_likelihood_
        c = self.model.concordance_index_

    def _train_cv(self, fold_index):
        # create lifelines dataset
        ds = pd.DataFrame(self.data.folds[fold_index].train.x.iloc[:,:self.params["input_size"]])
        ds["T"] = self.data.folds[fold_index].train.y["t"]
        ds["E"] = self.data.folds[fold_index].y["e"]
        CPH = CoxPHFitter(penalizer = self.params["wd"], l1_ratio = 0.)
        self.model = CPH.fit(ds, duration_col = "T", event_col = "E")
        l = self.model.log_likelihood_
        c = self.model.concordance_index_
        return c
    def _valid_cv(self, fold_index):
        test_data = self.data.folds[fold_index].test
        test_features = test_data.x
        test_t = test_data.y["t"]
        test_e = test_data.y["e"]
        out = self.model.predict_log_partial_hazard(test_features)
        l = self.loss(out, test_t, test_e)
        c = functions.compute_c_index(test_t, test_e, out)
        return out, l, c
        
    def set_fixed_params(self, hp_dict):

        self.params = hp_dict

    def _test(self, test_data):
        test_features = test_data.x
        test_t = test_data.y["t"]
        test_e = test_data.y["e"]
        out = self.model.predict_log_partial_hazard(test_features)
        l = self.loss(out, test_t, test_e)
        c = functions.compute_c_index(test_t, test_e, out)
        return out, l, c
    
    def loss(self, out, T, E): 
        return 999 


class CPHDNN(nn.Module):
    def __init__(self, data, nepochs = 1):
        super(CPHDNN, self).__init__()
        self.data = data
        self.params = DefaultDict()
        self.params["device"] = "cuda:0"
        self.params["crossval_nfolds"] = 5
        self.params["epochs"] = nepochs
        self.params["opt_nepochs"] = 1000
        self.params["lr"] = 1e-5
        self.params["c_index_cross_val"] = 0
        self.params["c_index_training"] = 0
        self.params["machine"] = os.uname()[1]
        self.params["process_id"] = os.getpid() 
        self.cols = ["process_id", "crossval_nfolds", "lr", "epochs","input_size","nInPCA",  "wd", "W", "D", "nL","c_index_training", "c_index_cross_val", "cpt_time"]    

        
        self.loss_training = []
        self.loss_valid = []
        self.c_index_training = []
        self.c_index_valid = []
    
    def set_fixed_params(self, hp_dict):

        self.params = hp_dict
        self.setup_stack()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.params["lr"], weight_decay = self.params["wd"])        

    def set_random_params(self):
        # weight decay or L2
        self.params["input_size"] = self.data.folds[0].train.x.shape[1] # dataset dependent!
        self.params["wd"] = np.power(10, np.random.uniform(-10, -9)) # V2 reasonable range for WD after analysis on V1 
        self.params["W"] = np.random.randint(3,2048) # V2 Reasonable
        self.params["D"] = np.random.randint(2,4) # V2 Reasonable
        # self.params["dp"] = np.random.uniform(0,0.5) # cap at 0.5 ! (else nans in output)
        self.params["nL"] = np.random.choice([nn.ReLU()]) 
        self.params["ARCH"] = {
            "W": np.concatenate( [[  ## ARCHITECTURE ###
            self.params["input_size"]], ### INSIZE
            np.ones(self.params["D"] - 1) * self.params["W"], ### N hidden = D - 1  
            [1]]).astype(int), ### OUTNODE 
            "nL": np.array([self.params["nL"] for i in range(self.params["D"])]),
            # "dp": np.ones(self.params["D"]) * self.params["dp"] 
        }
        self.params["nInPCA"] = np.random.randint(2,26)
        self.setup_stack()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.params["lr"], weight_decay = self.params["wd"])        

    def forward(self, x):
        risk = self.stack(x)
        return risk # torch.clamp(risk, min = -1000, max = 10)
    
    def loss(self, out, T, E): 
        uncensored_likelihood = torch.zeros(E.size())# list of uncensored likelihoods
        for x_i, E_i in enumerate(E): # cycle through samples
            if E_i == 1: # if uncensored ...
                log_risk = torch.log(torch.sum(torch.exp(out[:x_i +1])))
                uncensored_likelihood[x_i] = out[x_i] - log_risk # sub sum of log risks to hazard, append to uncensored likelihoods list
        
        loss = - uncensored_likelihood.sum() / (E == 1).sum() 
        return loss 
    
    def _train(self):
        bs = 24
        N = self.data.x.shape[0]
        self.nbatch = int(np.ceil(N / bs))
        self.data.to(self.params["device"])
        d = self.data
        for e in tqdm(range(self.params["opt_nepochs"]), desc="TRAINING FINAL MODEL"): # add timer 
            n = 0
            loss = 0
            c_index = 0
            for i in range(self.nbatch):
                train_ids = np.arange(i * bs , (i + 1) * bs)
                sorted_ids = torch.argsort(d.y[train_ids,0], descending = True) 
                train_features, train_T, train_E = d.x[sorted_ids], d.y[sorted_ids,0], d.y[sorted_ids,1]
                # train
                #print (f"train features: {train_features.size()}")
                #print (f"train T: {train_T.size()}")
                #print (f"train E: {train_E.size()}")
                #print(f"epoch: {e + 1} [{i+1}/{self.nbatch}]") 
                self.optimizer.zero_grad()
                try:  ### WORKAROUND, NANS in output of model 
                    out = self.forward(train_features)
                    l = self.loss(out, train_T, train_E)
                    if np.isnan(out.detach().cpu().numpy()).any():
                        raise ValueError("NaNs detected in forward pass")  ### WORKAROUND, NANS in output of model 
                    c = functions.compute_c_index(train_T.detach().cpu().numpy(), train_E.detach().cpu().numpy(), out.detach().cpu().numpy())
                    #print(f"c_index: {c}")
                    l.backward() 
        
                    self.optimizer.step()
                    loss += l
                    c_index += c
                    n += 1
                except ValueError:
                    return 0
                # print(f"loss: {loss/n}")
                # print(f"c_index: {c_index/n}")
            self.loss_training.append(loss.item() / n)
            self.c_index_training.append(c_index / n)
    def _train_cv(self, foldn):
        d =  self.data.folds[foldn].train
        
        bs = 24
        N = d.x.shape[0]
        self.nbatch = int(np.ceil(N / bs))
        for e in range(self.params["epochs"]): # add timer 
            n = 0
            loss = 0
            c_index = 0
            for i in range(self.nbatch):
                train_ids = np.arange(i * bs , (i + 1) * bs)
                sorted_ids = torch.argsort(d.y[train_ids,0], descending = True) 
                train_features, train_T, train_E = d.x[sorted_ids], d.y[sorted_ids,0], d.y[sorted_ids,1]
                # train
                #print (f"train features: {train_features.size()}")
                #print (f"train T: {train_T.size()}")
                #print (f"train E: {train_E.size()}")
                #print(f"epoch: {e + 1} [{i+1}/{self.nbatch}]") 
                self.optimizer.zero_grad()
                try:  ### WORKAROUND, NANS in output of model 
                    out = self.forward(train_features)
                    l = self.loss(out, train_T, train_E)
                    if np.isnan(out.detach().cpu().numpy()).any():
                        raise ValueError("NaNs detected in forward pass")  ### WORKAROUND, NANS in output of model 
                    c = functions.compute_c_index(train_T.detach().cpu().numpy(), train_E.detach().cpu().numpy(), out.detach().cpu().numpy())
                    #print(f"c_index: {c}")
                    l.backward() 
        
                    self.optimizer.step()
                    loss += l
                    c_index += c
                    n += 1
                except ValueError:
                    return 0
                # print(f"loss: {loss/n}")
                # print(f"c_index: {c_index/n}")
            self.loss_training.append(loss.item() / n)
            self.c_index_training.append(c_index / n)
            # test
            # for i, valid_data in enumerate(valid_dataloader):
            #     sorted_ids = torch.argsort(valid_data["t"], descending = True)
            #     valid_features, valid_T, valid_E = valid_data["data"][sorted_ids], valid_data["t"][sorted_ids], valid_data["e"][sorted_ids]
            #     l, c = self.test(valid_features, valid_T, valid_E)
            #     print(f"valid loss: {l}")
            #     print(f"valid c_index: {c}")
            #     self.loss_valid.append(l.item())
            #     self.c_index_valid.append(c)
    
    def _valid_cv(self, foldn):
        # forward prop
        # loss
        # c_index
        d = self.data.folds[foldn].test
        valid_features = d.x
        valid_t = d.y[:,0]
        valid_e = d.y[:,1]
        out = self.forward(valid_features)
        l = self.loss(out, valid_t, valid_e)
        c = functions.compute_c_index(valid_t.detach().cpu().numpy(),valid_e.detach().cpu().numpy(), out.detach().cpu().numpy())
        return out, l , c
    
    def _test(self, test_data):
        test_data.to(self.params["device"])
        test_features = test_data.x
        test_t = test_data.y[:,0]
        test_e = test_data.y[:,1]
        out = self.forward(test_features)
        l = self.loss(out, test_t, test_e)
        c = functions.compute_c_index(test_t.detach().cpu().numpy(), test_e.detach().cpu().numpy(), out.detach().cpu().numpy())
        return out, l, c

    def setup_stack(self):
        
        stack = []
        print("Setting up stack... saving to GPU")
        for layer_id in range(self.params["D"]):
            stack.append([
            [f"Linear_{layer_id}", nn.Linear(self.params["ARCH"]["W"][layer_id], self.params["ARCH"]["W"][layer_id + 1])],
            [f"Non-Linearity_{layer_id}", self.params["ARCH"]["nL"][0]]])            
        stack = np.concatenate(stack)
        stack = stack[:-1]# remove last non linearity !!
        self.stack = nn.Sequential(OrderedDict(stack)).to(self.params["device"])
    