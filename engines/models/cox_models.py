from torch import nn 
from collections import defaultdict, OrderedDict
from engines.hp_dict import base as HP_dict
import numpy as np
from lifelines import CoxPHFitter
import engines.models.functions as functions 
from experiments.plotting_functions import *
import pandas as pd
import os
import torch
from tqdm import tqdm
import pdb 
import matplotlib.pyplot as plt
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
        return {"l":l, "c":c}

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

    def _test(self, test_data, c_index = True):
        test_features = test_data.x
        test_t = test_data.y["t"]
        test_e = test_data.y["e"]
        out = self.model.predict_log_partial_hazard(test_features)
        l = None #= self.loss(out, test_t, test_e)
        c = functions.compute_c_index(test_t, test_e, out) if c_index else None
        return {"out":out, "l":l, "c":c}
    
    def loss(self, out, T, E): 
        # sort indices 
        pdb.set_trace()
        uncensored_likelihood = np.zeros(E.shape[0])# list of uncensored likelihoods
        for x_i, E_i in enumerate(E): # cycle through samples
            if E_i == 1: # if uncensored ...
                log_risk = np.log(np.sum(np.exp(out[:x_i +1])))
                uncensored_likelihood[x_i] = out[x_i] - log_risk # sub sum of log risks to hazard, append to uncensored likelihoods list
        
        loss = - uncensored_likelihood.sum() / (E == 1).sum() 
        return loss 

class ridge_CPHDNN(CPH):
    def __init__(self, data, modeltype="cphdnn", nepochs = 1):
        super(ridge_CPHDNN, self).__init__(data,nepochs)
        # init ridge_cph specs within CPHDNN framework
        self.hp_dict = HP_dict.generate_default(modeltype, data)

    def train(self):
        for foldn in range(self.hp_dict["nfolds"]):
            self.model = CPHDNN(self.data, hp_dict = self.hp_dict)
            print(self.model.params)
            train_c_index = self.model._train_cv(foldn)
            train_metrics = {"loss": self.model.loss_training, "c_index": self.model.c_index_training}
            out, l , c = self.model._valid_cv(foldn)
            valid_metrics = {"out":out, "loss": l, "c_index": c}
            print("tr_metrics:", train_c_index)
            print("vld_metrics:", l, c)
            plot_training(train_metrics["loss"], train_metrics["c_index"], foldn, self.params["modeltype"])
            
        return {"train":train_metrics, "valid": valid_metrics}

class CPHDNN(nn.Module):
    def __init__(self, data, hp_dict):
        super(CPHDNN, self).__init__()
        self.data = data
        self.params = hp_dict
        self.setup_stack(linear = self.params["linear"])
        self.optimizer = torch.optim.Adam(self.parameters(),  lr = self.params["lr"], weight_decay = self.params["wd"])
        # bunch of loggers 
        self.loss_training = []
        self.loss_valid = []
        self.c_index_training = []
        self.c_index_valid = []
    
     
    def _set_fixed_params(self):
        """
        deprecated.
        """
        self.params = defaultdict()
        self.params["input_size"] = self.data.x.shape[1]
        self.setup_stack()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.params["lr"], weight_decay = self.params["wd"])        
        self.params["machine"] = os.uname()[1]
        self.params["process_id"] = os.getpid() 


    def _random_init_params(self):
        self.params = defaultdict()
        # weight decay or L2
        self.params["input_size"] = self.data.folds[0].train.x.shape[1] # dataset dependent!
        self.params["wd"] = np.power(10, np.random.uniform(-10, -1)) # V2 reasonable range for WD after analysis on V1 
        self.params["W"] = 143 # np.random.randint(3,2048) # V2 Reasonable
        self.params["D"] = 2 # np.random.randint(2,4) # V2 Reasonable
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
        self.params["device"] = "cuda:0"
        self.params["lr"] = 1e-4
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
        """
        deprecated.
        """
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
        in_features =  self.data.folds[foldn].train # actual training features 
        nb_samples = in_features.x.shape[0] 
        mb_size =  nb_samples # mini-batch size (mb_size = nb-samples : gradient descent)
        self.nbatch = int(np.ceil(nb_samples / mb_size)) # nb mini-batches
        # load features and targets to GPU
        X = torch.Tensor(in_features.x.values).to("cuda:0")
        Y = torch.Tensor(in_features.y.values).to("cuda:0")
        
        #pdb.set_trace()
        for epoch_id in tqdm(range(self.params["nepochs"]), desc = f"fold {foldn}"): # add timer 
            # setup counters 
            nb_passes = 0
            total_loss = 0
            c_index = 0
            # loop through mini-batches
            for mbatch_id in range(self.nbatch):
                # sort samples for loss computations
                train_ids = np.arange(mbatch_id * mb_size , min((mbatch_id + 1) * mb_size, nb_samples))
                sorted_ids = torch.argsort(Y[train_ids,0], descending = True) 
                train_features, train_T, train_E = X[sorted_ids], Y[sorted_ids,0], Y[sorted_ids,1]
                #print(mbatch_id, nb_passes, total_loss, c_index, train_features.size())
                
                # train
                #print (f"train features: {train_features.size()}")
                #print (f"train T: {train_T.size()}")
                #print (f"train E: {train_E.size()}")
                #print(f"epoch: {e + 1} [{i+1}/{self.nbatch}]") 
                self.optimizer.zero_grad()
                out = self.forward(train_features)
                l = self.loss(out, train_T, train_E)
                c = functions.compute_c_index(train_T.detach().cpu().numpy(), train_E.detach().cpu().numpy(), out.detach().cpu().numpy())
                #print(f"c_index: {c}")
                l.backward() 
    
                self.optimizer.step()
                total_loss += float(l)
                c_index += c
                nb_passes += 1
                
                # print(f"loss: {loss/n}")
                # print(f"c_index: {c_index/n}")
            self.loss_training.append(total_loss / nb_passes)
            self.c_index_training.append(c_index / nb_passes)
            # test
            # for i, valid_data in enumerate(valid_dataloader):
            #     sorted_ids = torch.argsort(valid_data["t"], descending = True)
            #     valid_features, valid_T, valid_E = valid_data["data"][sorted_ids], valid_data["t"][sorted_ids], valid_data["e"][sorted_ids]
            #     l, c = self.test(valid_features, valid_T, valid_E)
            #     print(f"valid loss: {l}")
            #     print(f"valid c_index: {c}")
            #     self.loss_valid.append(l.item())
            #     self.c_index_valid.append(c)
        return c_index / nb_passes
         
    def _valid_cv(self, foldn):
       
        # forward prop
        # loss
        # c_index
        in_features =  self.data.folds[foldn].test
        valid_features_X = torch.Tensor(in_features.x.values).to("cuda:0")
        valid_features_Y = torch.Tensor(in_features.y.values).to("cuda:0")
        valid_t = valid_features_Y[:,0]
        valid_e = valid_features_Y[:,1]
        out = self.forward(valid_features_X)
        l = self.loss(out, valid_t, valid_e)
        c = functions.compute_c_index(valid_t.detach().cpu().numpy(),valid_e.detach().cpu().numpy(), out.detach().cpu().numpy())
    
        return out.detach().cpu().numpy(), l , c
    
    def _test(self, test_data):
        test_data.to(self.params["device"])
        test_features = test_data.x
        test_t = test_data.y[:,0]
        test_e = test_data.y[:,1]
        out = self.forward(test_features)
        l = self.loss(out, test_t, test_e)
        c = functions.compute_c_index(test_t.detach().cpu().numpy(), test_e.detach().cpu().numpy(), out.detach().cpu().numpy())
        return out.detach().cpu().numpy(), l, c

    def setup_stack(self, linear = False):
        if linear:
            self.stack = nn.Linear(self.params["input_size"], 1).to(self.params["device"]) 
            return
        stack = []
        print("Setting up stack... saving to GPU")
        
        ## input layer, Hidden 1
        stack.append([
            [f"Linear_0", nn.Linear(self.params["input_size"], self.params["W"])],
            [f"Non-Linearity_0", self.params["nL"]]])             
        
        ## hidden layers
        for layer_id in range(self.params["D"]-1):
            stack.append([
            [f"Linear_{layer_id+1}", nn.Linear(self.params["W"], 1)]])            
        
        ## output layer
        #stack.append([
        #   [f"Linear_{layer_id + 1}", nn.Linear(self.params["W"], 1)],
        #   [f"Non-Linearity_{layer_id + 1}", self.params["nL"]]])
        
        stack = np.concatenate(stack)
        # remove last non-lin (do not need it)
        self.stack = nn.Sequential(OrderedDict(stack)).to(self.params["device"])
    