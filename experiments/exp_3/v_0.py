from engines.datasets.base_datasets import SurvivalGEDataset
from engines.models import cox_models
import matplotlib.pyplot as plt
import numpy as np
import pdb 
import pandas as pd

def train_cphdnn(train_data, nfolds = 5):  
    model = cox_models.ridge_CPHDNN(train_data, modeltype="cphdnn")
    tr_metrics = model.train()
    return model, tr_metrics["train"], tr_metrics["valid"]

def train_ridge_cph(train_data, nfolds = 5):
    model = cox_models.ridge_CPHDNN(train_data, modeltype="ridge_cph")
    tr_metrics = model.train()
    return model, tr_metrics["train"], tr_metrics["valid"] 

def run(args):
    print("loading data")
    data_types = []
    SGE = SurvivalGEDataset()
    SGE.get_data(args.COHORT)
    # take CF + LSC17
    CF, LSC17 = SGE.data["CF_bin"], SGE.data["LSC17"]
    CF_LSC17 = SGE.new(
        x = pd.concat([SGE.data["CF_bin"], SGE.data["LSC17"].x], axis = 1), 
        y = SGE.data["LSC17"].y)
    data_types.append(CF_LSC17)
    # take CDS
    # CDS_HIEXP
    # take PCA300 (loadings only)
    # PCA_300
    ## Takes input 
    # get full transcr. profiles 
    #data = cohort_data["CDS"]
    #data.generate_PCA(input_size = 300)
    for data in data_types :
        data.split_train_test(args.NFOLDS)
        # trains ridge cph
        #model, tr_loss, tr_c_index = train_ridge_cph(data, nfolds = args.NFOLDS)

        model, tr_metrics, vld_metrics = train_cphdnn(data, nfolds = args.NFOLDS)
        np.sort(vld_metrics["c_index"])
        c95 = vld_metrics["c_index"][int(1000*0.95)]
        c05 = vld_metrics["c_index"][int(1000*0.05)]
        print("final c index CPHDNN (boostrapped):" , np.median(vld_metrics["c_index"]), c95, c05)
        pdb.set_trace()
        """ model, tr_metrics, vld_metrics = train_ridge_cph(data, nfolds = args.NFOLDS)
        c95 = vld_metrics["c_index"][int(1000*0.95)]
        c05 = vld_metrics["c_index"][int(1000*0.05)]
        print("final c index CPH (boostrapped):" , round(np.median(vld_metrics["c_index"]), 3),  round(c05,3), round(c95,3))
         """
    # report training (c_index, loss)
    #plot_data(train_data)
    # tests
    #test_model(model, test_data)
    # reports (c_index, loss)  
    # print results  

