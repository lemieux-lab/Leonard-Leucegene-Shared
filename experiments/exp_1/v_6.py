from engines.utils import assert_mkdir
from experiments.plotting_functions import *
from engines.models.functions import *
from engines.functions import * 
from engines.datasets.base_datasets import SurvivalGEDataset
from engines.models import cox_models
from engines.hp_dict.base import HP_dict
from collections import Counter
import os
import pdb 

def run(args):

    assert_mkdir(args.OUTPATH)
    lgn_pronostic = SurvivalGEDataset().get_data("lgn_pronostic")
    LGN_CDS = lgn_pronostic["CDS"]
    LGN_LSC17 = lgn_pronostic["LSC17"]
    LGN_CF = lgn_pronostic["CF_bin"]
    
    lgn_int = SurvivalGEDataset().get_data("lgn_intermediate")

    LGN_INT_CDS = lgn_int["CDS"]
    LGN_INT_LSC17 = lgn_int["LSC17"]
    tcga_aml = SurvivalGEDataset().get_data("tcga_target_aml")
    TCGA_CDS = tcga_aml["CDS"]
    TCGA_LSC17 = tcga_aml["LSC17"]
    lgn_cyt_levels = [{"intermediate cytogenetics":1, "adverse cytogenetics": 2, "favorable cytogenetics":0 }[level] for level in lgn_pronostic["CF"]["Cytogenetic risk"]]
    tcga_cyt_levels = [{"Standard":1, "Low": 2, "Favorable":0 }[level] for level in tcga_aml["CF"]["Cytogenetic risk"]]



    LGN_LSC17.split_train_test(args.NFOLDS)
    LGN_CDS.split_train_test(args.NFOLDS)
    LGN_INT_LSC17.split_train_test(args.NFOLDS)
    LGN_INT_CDS.split_train_test(args.NFOLDS)
    TCGA_LSC17.split_train_test(args.NFOLDS)
    TCGA_CDS.split_train_test(args.NFOLDS)
    
    HyperParams = HP_dict(args.WEIGHT_DECAY, args.NEPOCHS, args.bootstr_n, args.NFOLDS)
    LGN_LSC17_params = HyperParams.generate_default("ridge_cph_lifelines_LSC17", LGN_LSC17)        
    LGN_PCA17_params = HyperParams.generate_default("ridge_cph_lifelines_PCA", LGN_CDS) 
    LGN_PCA_PARAMS = {"min_col": 0, "max_col": LGN_CDS.x.shape[1], "pca_n": 17}       
    LGN_INT_LSC17_params = HyperParams.generate_default("ridge_cph_lifelines_LSC17", LGN_INT_LSC17)        
    LGN_INT_PCA17_params = HyperParams.generate_default("ridge_cph_lifelines_PCA", LGN_INT_CDS)        
    LGN_INT_PCA_PARAMS = {"min_col": 0, "max_col": LGN_INT_CDS.x.shape[1], "pca_n": 17}   
    TCGA_LSC17_params = HyperParams.generate_default("ridge_cph_lifelines_LSC17", TCGA_LSC17)        
    TCGA_PCA17_params = HyperParams.generate_default("ridge_cph_lifelines_PCA", TCGA_CDS)   
    TCGA_PCA_PARAMS = {"min_col": 0, "max_col": TCGA_CDS.x.shape[1], "pca_n": 17}   
    
    # NO CF  
    

    # WITH CF
    plot_c_surv_3_groups(cox_models.evaluate(TCGA_LSC17, TCGA_LSC17_params, pca_params = TCGA_PCA_PARAMS)[2], TCGA_LSC17_params, args.OUTPATH, group_weights = Counter(tcga_cyt_levels))
    plot_c_surv_3_groups(cox_models.evaluate(TCGA_CDS, TCGA_PCA17_params, pca_params = TCGA_PCA_PARAMS)[2], TCGA_PCA17_params,args.OUTPATH, group_weights = Counter(tcga_cyt_levels))
    
    # OK 
    plot_c_surv_3_groups(cox_models.evaluate(LGN_LSC17, LGN_LSC17_params)[2],  LGN_LSC17_params, args.OUTPATH, group_weights = Counter(lgn_cyt_levels))
    plot_c_surv_3_groups(cox_models.evaluate(LGN_CDS, LGN_PCA17_params, pca_params = LGN_PCA_PARAMS)[2],LGN_PCA17_params, args.OUTPATH, group_weights = Counter(lgn_cyt_levels))
    plot_c_surv(cox_models.evaluate(LGN_INT_LSC17, LGN_INT_LSC17_params), args.OUTPATH, group_weights = Counter(lgn_cyt_levels))
    plot_c_surv(cox_models.evaluate(LGN_INT_CDS, LGN_INT_PCA17_params, pca_params = LGN_INT_PCA_PARAMS), args.OUTPATH, group_weights = Counter(lgn_cyt_levels))
    
    
    pdb.set_trace()    
    
    plot_c_surv_3_groups(cox_models.evaluate(LGN_CF_LSC17, LGN_LSC17_params)[2],  LGN_LSC17_params, args.OUTPATH, group_weights = Counter(lgn_cyt_levels))
    plot_c_surv_3_groups(cox_models.evaluate(LGN_CF_CDS, LGN_PCA17_params, pca_params = LGN_PCA_PARAMS)[2],LGN_PCA17_params, args.OUTPATH, group_weights = Counter(lgn_cyt_levels))
    
    # TO DO
    # plot_c_surv(cox_models.evaluate(LGN_INT_LSC17, LGN_INT_LSC17_params), args.OUTPATH, group_weights = Counter(cyt_levels))
    # plot_c_surv(cox_models.evaluate(LGN_INT_CDS, LGN_INT_PCA17_params, pca_params = LGN_INT_PCA_PARAMS), args.OUTPATH, group_weights = Counter(cyt_levels))
    

    pdb.set_trace()
    
    