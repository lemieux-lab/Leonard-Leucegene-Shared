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
from sklearn.decomposition import PCA 

# METHODS
def evaluate(model_type, data, HyperParams, pca_n = None):
    # split training / valid
    data.split_train_test(HyperParams.nfolds)
    # generate Hyper-Parameter dict specific to model type
    params = HyperParams.generate_default(model_type, data)
    # correct input size
    if pca_n : params["input_size"] = pca_n
    # instanciate new model container 
    model = CPH(model_type, data, params, pca_n)
    # train model by cross_validation
    metrics = model.cross_validation()
    # bootstrap c index and return scores
    return metrics 

class ridge_cph_lifelines:
    def __init__(self, params) -> None:
        self.params = params
        self.model = CoxPHFitter(penalizer = self.params["wd"], l1_ratio = 0.)

    def _train(self, data):    
        self.train_ds = pd.DataFrame(data.x.iloc[:,:self.params["input_size"]])
        self.train_ds["T"] = data.y["t"]
        self.train_ds["E"] = data.y["e"]
        self.model = self.model.fit(self.train_ds, duration_col = "T", event_col = "E")
    
    def _valid(self, data):
        self.vld_features= data.x
        self.vld_t = data.y["t"]
        self.vld_e = data.y["e"]
        out = self.model.predict_log_partial_hazard(self.vld_features)
        return out 

class ridge_cph:
    def __init__(self, data, modeltype="cphdnn", nepochs = 1):
        # init ridge_cph specs within CPHDNN framework
        self.hp_dict = HP_dict.generate_default(modeltype, data)
    
    def _train(self):
        pass
    def _valid(self):        
        pass 

# CPH container class       
class CPH:
    def __init__(self, model_type, data, params, pca_n):
        picker = {"ridge_cph_lifelines": ridge_cph_lifelines, 
        "cphdnn": CPHDNN}
        self.model = picker[model_type](params)
        self.data = data
        self.params = params
        self.pca_n = pca_n

    def cross_validation(self):
        vld_scores = []        
        for foldn in tqdm(range(self.params["nfolds"]), desc = f"{self.params['modeltype']} INsize: {self.params['input_size']}"):
            # get PCA loadings for input transf. if needed 
            pca = functions.compute_pca_loadings(self.data.folds[foldn].train.x, self.pca_n)
            # train model on fold
            train_c_index = self._train_on_fold(foldn, transform_input = pca)
            # inference on foldn vld set
            out = self._valid_on_fold(foldn, transform_input = pca)
            #print("tr_metrics:", train_c_index)
            #print("vld_metrics:", float(l), c)
            #plot_training(train_metrics["loss"], train_metrics["c_index"], foldn, self.hp_dict["modeltype"])
            vld_scores.append(out)
        vld_scores = np.concatenate(vld_scores)
        c_scores, metrics = functions.compute_aggregated_bootstrapped_c_index(vld_scores, self.data.y, n = self.params["bootstrap_n"])
        print(metrics)
        return c_scores 

    def _train_on_fold(self, fold_index, transform_input):
        # gets current fold data
        fold_train_data = self.data.folds[fold_index].train
        # transforms according to loadings
        if transform_input is not None: 
            new_df = pd.DataFrame(np.dot(fold_train_data.x.values - transform_input["mean"], transform_input["components"].T), index = fold_train_data.x.index)
            fold_train_data.x = new_df
        # performs training of model
        self.model._train(fold_train_data)
        
    def _valid_on_fold(self, fold_index, transform_input):
        # gets current fold data
        fold_vld_data = self.data.folds[fold_index].test
        # transforms according to loadings
        if transform_input is not None: 
            new_df = pd.DataFrame(np.dot(fold_vld_data.x.values - transform_input["mean"], transform_input["components"].T), index = fold_vld_data.x.index)
            fold_vld_data.x = new_df
        # inference
        out = self.model._valid(fold_vld_data)
        return out

class CPHDNN(nn.Module):
    def __init__(self, hp_dict):
        super(CPHDNN, self).__init__()
        self.params = hp_dict
        self.setup_stack(linear = self.params["linear"])
        self.optimizer = torch.optim.Adam(self.parameters(),  lr = self.params["lr"], weight_decay = self.params["wd"])
        # bunch of loggers 
        self.loss_training = []
        self.loss_valid = []
        self.c_index_training = []
        self.c_index_valid = []
    
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

    def _train(self, in_features):
        # actual training features 
        nb_samples = in_features.x.shape[0] 
        mb_size =  nb_samples # mini-batch size (mb_size = nb-samples : gradient descent)
        self.nbatch = int(np.ceil(nb_samples / mb_size)) # nb mini-batches
        # load features and targets to GPU
        X = torch.Tensor(in_features.x.values).to("cuda:0")
        Y = torch.Tensor(in_features.y.values).to("cuda:0")
        
        #pdb.set_trace()
        for epoch_id in range(self.params["nepochs"]): # add timer 
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
         
    def _valid(self, in_features):
       
        valid_features_X = torch.Tensor(in_features.x.values).to("cuda:0")
        valid_features_Y = torch.Tensor(in_features.y.values).to("cuda:0")
        valid_t = valid_features_Y[:,0]
        valid_e = valid_features_Y[:,1]
        out = self.forward(valid_features_X)
        l = self.loss(out, valid_t, valid_e)
        c = functions.compute_c_index(valid_t.detach().cpu().numpy(),valid_e.detach().cpu().numpy(), out.detach().cpu().numpy())
        return out.detach().cpu().numpy()
    
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
        print("Setting up stack... saving to GPU")
        # for the linear model
        if linear:
            self.stack = nn.Linear(self.params["input_size"], 1).to(self.params["device"]) 
            return
        
        # for the MLP model
        stack = []
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
    