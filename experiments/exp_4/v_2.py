from curses.ascii import SUB
from datetime import datetime 
import os 
from engines import utils
from engines.datasets.base_datasets import SurvivalGEDataset, Data
from engines.models import functions 
from engines.models import cox_models 
from engines.hp_dict.base import HP_dict
from lifelines import KaplanMeierFitter
from sklearn.metrics import confusion_matrix
from collections import Counter
import numpy as np
import pandas as pd
import pdb 
import matplotlib.pyplot as plt
from experiments.plotting_functions import *
plt.rcParams["svg.fonttype"] = "none" # text rendering in figures output 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def run(args):
    SGE = SurvivalGEDataset()
    ## cohort ## input_types ## other params
    ## data{x: input_data, y: target} 
    ## Clinical FEATURES 
    mutations = ["NPM1 mutation", "FLT3-ITD mutation", "IDH1-R132 mutation"]
    age_sex = ["Sex_F","Age_gt_60"]
    cytogenetics = ['MLL translocations (+MLL FISH positive) (Irrespective of additional cytogenetic abnormalities)',
        'Intermediate abnormal karyotype (except isolated trisomy/tetrasomy 8)',
        'Normal karyotype',
        'Complex (3 and more chromosomal abnormalities)',
        'Trisomy/tetrasomy 8 (isolated)',
        'Monosomy 5/ 5q-/Monosomy 7/ 7q- (less than 3 chromosomal abnormalities)',
        'NUP98-NSD1(normal karyotype)',
        't(8;21)(q22;q22)/RUNX1-RUNX1T1 (Irrespective of additional cytogenetic abnormalities)',
        'inv(16)(p13.1q22)/t(16;16)(p13.1;q22)/CBFB-MYH11 (Irrespective of additional cytogenetic abnormalities)',
        'EVI1 rearrangements (+EVI1 FISH positive) (Irrespective of additional cytogenetic abnormalities)',
        't(6;9)(p23;q34) (Irrespective of additional cytogenetic abnormalities)',
        'Monosomy17/del17p (less than 3 chromosomal abnormalities)',
        'Hyperdiploid numerical abnormalities only']
    clinical_features = np.concatenate([mutations, cytogenetics, age_sex])
    SGE.get_data("lgn_pronostic")
    ## cohort ## input_types ## other params
    ## data{x: input_data, y: target} 
    clin_factors = SGE.new(clinical_features, gene_expressions="None")
    clin_factors_lsc17 = SGE.new(clinical_features, gene_expressions="LSC17")
    clin_factors_pca = SGE.new(clinical_features, gene_expressions="PCA")
    
    pca_only = SGE.new(None, gene_expressions = "PCA")
    lsc17_only = SGE.new(None, gene_expressions = "LSC17")
    lsc17_pca =  SGE.new(None, gene_expressions = "LSC17+PCA")

    clin_factors_lsc17_pca = SGE.new(clinical_features, gene_expressions="LSC17+PCA")

    #clin_factors = SGE.new("lgn_pronostic", clinical_features, gene_expressions="None")
    #clin_factors_lsc17 = SGE.new("lgn_pronostic", clinical_features, gene_expressions="LSC17")

    # Set general parameters
    HyperParams = HP_dict(wd = 1e-3, nepochs = 200,  bootstr_n = 1000, nfolds = 5)

    # compute cytogentic risk benchmark
    CDS = SGE.get_data("lgn_pronostic")["CDS"]
    cyt = pd.DataFrame(SGE.data["CF"]["Cytogenetic risk"])
    cyt_levels = [{"intermediate cytogenetics":1, "Intermediate/Normal":1, "adverse cytogenetics": 2, "favorable cytogenetics":0, "Favorable":0, "Standard":1, "Low":0, "Poor":2, None: 1}[level] for level in cyt["Cytogenetic risk"]] 
    cyt["pred_risk"] = cyt_levels
    cyt_c_scores, cyt_metrics = functions.compute_cyto_risk_c_index(cyt["pred_risk"], CDS.y, gamma = 0.001, n = HyperParams.bootstr_n)
    print("C index method 1: ", cyt_metrics)

    results = [(1, "c_index", "cytogenetics", cyt_metrics[0], cyt_metrics[1], cyt_metrics[2] )]

    # for repn in range(1,4,1):   
    #     for model_type in ["ridge_cph_lifelines_CF_PCA", "cphdnn_1l"]:
    #         data = clin_factors_pca.clone()
    #         # preprocess data 
    #         data.x = data.x[data.x.columns[np.where(data.x.var(0) > 0.01)]]
    #         # splitting
    #         data.split_train_test(HyperParams.nfolds)
    #         # generate model parameters 
    #         params = HyperParams.generate_default(model_type = model_type, data = data)
    #         pca_params = {"min_col": 16, "max_col": data.x.shape[1], "pca_n": 50 }
    #         # train and evaluate model
    #         c_index_metrics, c_scores, surv_tbl, params= cox_models.evaluate(data, params, pca_params = pca_params)
    #         # append to results
    #         results.append((repn, params["modeltype"], data.name , c_index_metrics[0], c_index_metrics[1], c_index_metrics[2] ))

    for repn in range(1,4,1):   
        for model_type in ["ridge_cph_lifelines_CF_PCA", "cphdnn_6l"]:
            data = clin_factors_lsc17_pca.clone()
            # preprocess data 
            var = data.x.var(0)
            data.x = data.x[data.x.columns[np.where( var > 0.01)]]
            # splitting
            data.split_train_test(HyperParams.nfolds)
            # generate model parameters 
            params = HyperParams.generate_default(model_type = model_type, data = data)
            pca_params = {"min_col": 33, "max_col": data.x.shape[1], "pca_n": 25 }
            # train and evaluate model
            c_index_metrics, c_scores, surv_tbl, params= cox_models.evaluate(data, params, pca_params = pca_params)
            # append to results
            results.append((repn, params["modeltype"], data.name , c_index_metrics[0], c_index_metrics[1], c_index_metrics[2] ))
     
    for repn in range(1,4,1):   
        for model_type in ["ridge_cph_lifelines_CF_PCA", "cphdnn_1l"]:
            data = lsc17_pca.clone()
            # preprocess data 
            var = data.x.var(0)
            data.x = data.x[data.x.columns[np.where( var > 0.01)]]
            # splitting
            data.split_train_test(HyperParams.nfolds)
            # generate model parameters 
            params = HyperParams.generate_default(model_type = model_type, data = data)
            pca_params = {"min_col": 17, "max_col": data.x.shape[1], "pca_n": 50 }
            # train and evaluate model
            c_index_metrics, c_scores, surv_tbl, params= cox_models.evaluate(data, params, pca_params = pca_params)
            # append to results
            results.append((repn, params["modeltype"], data.name , c_index_metrics[0], c_index_metrics[1], c_index_metrics[2] ))
    
    for repn in range(1,4,1):   
        for model_type in ["ridge_cph_lifelines_CF_PCA", "cphdnn_1l"]:
            data = pca_only.clone()
            # preprocess data 
            var = data.x.var(0)
            data.x = data.x[data.x.columns[np.where( var > np.median(var))]]
            # splitting
            data.split_train_test(HyperParams.nfolds)
            # generate model parameters 
            params = HyperParams.generate_default(model_type = model_type, data = data)
            pca_params = {"min_col": 0, "max_col": data.x.shape[1], "pca_n": 50 }
            # train and evaluate model
            c_index_metrics, c_scores, surv_tbl, params= cox_models.evaluate(data, params, pca_params = pca_params)
            # append to results
            results.append((repn, params["modeltype"], data.name , c_index_metrics[0], c_index_metrics[1], c_index_metrics[2] ))
