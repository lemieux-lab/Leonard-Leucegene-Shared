from engines.datasets.base_datasets import SurvivalGEDataset
from engines.models import cox_models
from engines.hp_dict.base import HP_dict
import matplotlib.pyplot as plt
import numpy as np
import pdb 
import pandas as pd
from engines.models import functions 
from datetime import datetime

def run(args):
    print("loading data")
    # get timestamp
    timestamp = "".join(str(datetime.now()).split(".")[:-1]).replace(" ", "_").replace("-", "_").replace(".", "_").replace(":","_")
    data_types = []
    SGE = SurvivalGEDataset()
    SGE.get_data(args.COHORT)
    ## PREPARE INPUT DATA TYPES
    # LSC17 gene signature
    LSC17 = SGE.data["LSC17"]
    # take CF + LSC17
    CF, LSC17 = SGE.data["CF_bin"], SGE.data["LSC17"]
    CF_LSC17 = SGE.new(
        x = pd.concat([CF, LSC17.x], axis = 1), 
        y = SGE.data["LSC17"].y)
    # data_types.append(CF_LSC17)
    # take CDS
    CDS = SGE.data["CDS"]
    # input fixed hyper-params
    HyperParams = HP_dict(args)
    scores_matrix = []
    # compute c index on cyt risk
    cyt = pd.DataFrame(SGE.data["CF"]["Cytogenetic risk"])
    cyt_levels = [{"intermediate cytogenetics":1, "adverse cytogenetics": 2, "favorable cytogenetics":0 }[level] for level in cyt["Cytogenetic risk"]] 
    cyt["levels"] = cyt_levels
    c_ind = functions.compute_c_index(CDS.y["t"], CDS.y["e"], cyt.levels)
    cyt_c_scores, cyt_metrics  = functions.compute_aggregated_bootstrapped_c_index(cyt.levels, CDS.y, n=args.bootstr_n)
    #print(cyt_metrics)
    scores_matrix.append(("cyto.risk_scores", cyt_c_scores))
    # take PCA300 (loadings only)
    for model in args.MODEL_TYPES:
        scores_matrix.append((f"LSC17_{model}_scores", cox_models.evaluate(model, LSC17, HyperParams)))
        scores_matrix.append((f"CDS_{model}_scores", cox_models.evaluate(model, CDS, HyperParams)))
        for redux_size in [25]:
            scores_matrix.append((f"PCA_{redux_size}_{model}_scores", cox_models.evaluate(model, CDS, HyperParams, pca_n = redux_size)))
    
            scores = pd.DataFrame(dict(scores_matrix))
            scores.to_csv(f"RES/POSTER_RECOMB/data/{timestamp}_results.txt")
    ## Takes input 
    # get full transcr. profiles 
    pdb.set_trace()
    #data = cohort_data["CDS"]
    #data.generate_PCA(input_size = 300)
    """ for data in data_types :
        data.split_train_test(args.NFOLDS)
        # trains ridge cph
        # model, tr_loss, tr_c_index = train_ridge_cph(data, nfolds = args.NFOLDS)

        model, tr_metrics, vld_metrics = train_cphdnn(data, nfolds = args.NFOLDS, bootstr_n = args.bootstr_n)
        print("final c index CPHDNN (boostrapped):" , vld_metrics["c_ind_med"], f"({vld_metrics['c_ind_min']},{vld_metrics['c_ind_max']})")
        pdb.set_trace()
        model, tr_metrics, vld_metrics = train_ridge_cph(data, nfolds = args.NFOLDS)
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

