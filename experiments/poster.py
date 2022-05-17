from engines.utils import assert_mkdir
from experiments.plotting_functions import *
from engines.models.functions import *
from engines.functions import * 
from engines.datasets.base_datasets import SurvivalGEDataset
from engines.models import cox_models
from engines.hp_dict.base import HP_dict
import os
import pdb 

def run(args):
    # computations 
    assert_mkdir(args.OUTPATH)
    lgn_pronostic = SurvivalGEDataset().get_data("lgn_pronostic")
    LGN_CDS = lgn_pronostic["CDS"]
    LGN_LSC17 = lgn_pronostic["LSC17"]
    LGN_CF = lgn_pronostic["CF"]
    LGN_PCA17 = PCA_transform(LGN_CDS.x, 17)
    lgn_int = SurvivalGEDataset().get_data("lgn_intermediate")
    LGN_INT_CDS = lgn_int["CDS"]
    LGN_INT_LSC17 = lgn_int["LSC17"]
    tcga_aml = SurvivalGEDataset().get_data("tcga_target_aml")
    TCGA_CDS = tcga_aml["CDS"]
    TCGA_LSC17 = tcga_aml["LSC17"]
    # 1 output LSC17 genes
    #LGN_LSC17.to_csv(os.path.join(args.OUTPATH, "1_lsc17.csv"), index = False)
    # 2 T-SNE from PCA(17) Leucegene-All samples
    #LGN_TSNE_PCA17 = TSNE_transform(LGN_PCA17)
    #plot_tsne(LGN_TSNE_PCA17, LGN_CF,  "who 2008", args.OUTPATH)
    #plot_tsne(LGN_TSNE_PCA17, LGN_CF,  "cyto risk", args.OUTPATH)
    # plot_tsne(tsne_sample_x)
    # 3 Dimension / Method
    # 4 Overfitting
    #plot_overfit(rselect_overfit)
    #plot_overfit(rproj_overfit)
    #plot_overfit(CF_PCA_overfit)
    #plot_overfit(PCA)
    # 5 concordance / surv curves (hi-risk/low-risk)
    LGN_LSC17.split_train_test(args.NFOLDS)
    LGN_CDS.split_train_test(args.NFOLDS)
    LGN_INT_LSC17.split_train_test(args.NFOLDS)
    LGN_INT_CDS.split_train_test(args.NFOLDS)
    TCGA_LSC17.split_train_test(args.NFOLDS)
    TCGA_CDS.split_train_test(args.NFOLDS)
    
    HyperParams = HP_dict(args)
    LGN_LSC17_params = HyperParams.generate_default("ridge_cph_lifelines_LSC17", LGN_LSC17)        
    LGN_PCA17_params = HyperParams.generate_default("ridge_cph_lifelines_PCA", LGN_CDS)        
    LGN_INT_LSC17_params = HyperParams.generate_default("ridge_cph_lifelines_LSC17", LGN_INT_LSC17)        
    LGN_INT_PCA17_params = HyperParams.generate_default("ridge_cph_lifelines_PCA", LGN_INT_CDS)        
    TCGA_LSC17_params = HyperParams.generate_default("ridge_cph_lifelines_LSC17", TCGA_LSC17)        
    TCGA_PCA17_params = HyperParams.generate_default("ridge_cph_lifelines_PCA", TCGA_CDS)        
    #plot_c_surv(cox_models.evaluate(CF_LSC17, params))
    #plot_c_surv(cox_models.evaluate(CF_PCA))
    plot_c_surv(cox_models.evaluate(LGN_LSC17, LGN_LSC17_params), args.OUTPATH)
    plot_c_surv(cox_models.evaluate(LGN_CDS, LGN_PCA17_params, pca_n = 17), args.OUTPATH)
    plot_c_surv(cox_models.evaluate(LGN_INT_LSC17, LGN_INT_LSC17_params), args.OUTPATH)
    plot_c_surv(cox_models.evaluate(LGN_INT_CDS, LGN_INT_PCA17_params, pca_n = 17), args.OUTPATH)
    plot_c_surv(cox_models.evaluate(TCGA_LSC17, TCGA_LSC17_params ), args.OUTPATH)
    plot_c_surv(cox_models.evaluate(TCGA_CDS, TCGA_PCA17_params, pca_n = 17 ), args.OUTPATH)
    # 6 correlations
    #plot_correlations(LSC17, CF)
    #plot_correlations(PCA17, CF)
    #plot_variance(PCA17)
    # 7 Reclassification (surv curves cox-PH, separation between 3 groups)
    #plot_c_surv_3_groups(CYT)
    #plot_c_surv_3_groups(LSC17)
    #plot_c_surv_3_groups(PCA17)
    #plot_cm(LSC17, CYT)
    #plot_cm(PCA17, CYT)
