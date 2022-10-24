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
    # computations 
    assert_mkdir(args.OUTPATH)
    lgn_pronostic = SurvivalGEDataset().get_data("lgn_pronostic")
    LGN_CDS = lgn_pronostic["CDS"]
    LGN_LSC17 = lgn_pronostic["LSC17"]
    LGN_CF = lgn_pronostic["CF_bin"]
    LGN_PCA17 = PCA_transform(LGN_CDS.x, 17)
    lgn_int = SurvivalGEDataset().get_data("lgn_intermediate")

    LGN_INT_CDS = lgn_int["CDS"]
    LGN_INT_LSC17 = lgn_int["LSC17"]
    tcga_aml = SurvivalGEDataset().get_data("tcga_target_aml")
    TCGA_CDS = tcga_aml["CDS"]
    TCGA_LSC17 = tcga_aml["LSC17"]
    cyt_levels = [{"intermediate cytogenetics":1, "adverse cytogenetics": 2, "favorable cytogenetics":0 }[level] for level in lgn_pronostic["CF"]["Cytogenetic risk"]]
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
    
    #plot_c_surv_3_groups(cox_models.evaluate(LGN_LSC17, LGN_LSC17_params)[2],  LGN_LSC17_params, args.OUTPATH, group_weights = Counter(cyt_levels))
    #plot_c_surv_3_groups(cox_models.evaluate(LGN_CDS, LGN_PCA17_params, pca_params = LGN_PCA_PARAMS)[2],LGN_PCA17_params, args.OUTPATH, group_weights = Counter(cyt_levels))
    plot_c_surv(cox_models.evaluate(LGN_INT_LSC17, LGN_INT_LSC17_params), args.OUTPATH, group_weights = Counter(cyt_levels))
    plot_c_surv(cox_models.evaluate(LGN_INT_CDS, LGN_INT_PCA17_params, pca_params = LGN_INT_PCA_PARAMS), args.OUTPATH, group_weights = Counter(cyt_levels))
    #plot_c_surv_3_groups(cox_models.evaluate(TCGA_LSC17, TCGA_LSC17_params )[2], TCGA_LSC17_params, args.OUTPATH, group_weights = Counter(cyt_levels))
    #plot_c_surv_3_groups(cox_models.evaluate(TCGA_CDS, TCGA_PCA17_params, pca_params = TCGA_PCA_PARAMS)[2], TCGA_PCA17_params,args.OUTPATH, group_weights = Counter(cyt_levels))
    
    pdb.set_trace()
    # 6 correlations
    # plot heatmap of table --> heatmap_pca.svg
    features = ['Age_at_diagnosis', 'adverse cytogenetics', 
        'favorable cytogenetics', 'intermediate cytogenetics',
        'NPM1 mutation_1.0', 'IDH1-R132 mutation_1.0', 
        'FLT3-ITD mutation_1', 'Sex_F']
    #plot_correlations(get_corr_to_cf(LGN_LSC17.x, LGN_CF), features, "LSC17", args.OUTPATH, figsize = (12,12))
    #plot_correlations(get_corr_to_cf(LGN_PCA17, LGN_CF), features, "PCA17", args.OUTPATH, figsize = (12,12))
    # plot expl. var / var ratio vs #PC --> pca var .svg
    #pc_var_plot(corr_df)
    #plot_correlations(LSC17, CF)
    #plot_correlations(PCA17, CF)
    #plot_variance(PCA17)
    # 7 Reclassification (surv curves cox-PH, separation between 3 groups)
    SGE = SurvivalGEDataset()
    SGE.get_data("lgn_pronostic")
    CDS = SGE.data["CDS"]
    cyt = pd.DataFrame(SGE.data["CF"]["Cytogenetic risk"])
    cyt_levels = [{"intermediate cytogenetics":1, "adverse cytogenetics": 2, "favorable cytogenetics":0 }[level] for level in cyt["Cytogenetic risk"]] 
    cyt["pred_risk"] = cyt_levels
    cyt_c_scores_1, cyt_metrics_1 = compute_cyto_risk_c_index(cyt["pred_risk"],CDS.y, gamma = 0.001, n = args.bootstr_n)
    cyt_c_scores_2, cyt_metrics_2  = compute_aggregated_bootstrapped_c_index(cyt["pred_risk"], CDS.y, n=args.bootstr_n)
    params = HyperParams.generate_default("cytogenetic_risk", cyt["pred_risk"])
    params["c_index_metrics"] = cyt_metrics_1[0]          
    cyt["e"] = CDS.y["e"]
    cyt["t"] = CDS.y["t"]
    print("C index method 1: ", cyt_metrics_1)
    print("C index method 2: ", cyt_metrics_2)
    #plot_c_surv_3_groups(cyt, params, args.OUTPATH, group_weights = Counter(cyt_levels))
    c_index_metrics, c_scores, lsc17_surv_tbl, lsc17_params, model = cox_models.evaluate(LGN_LSC17, LGN_LSC17_params)
    #plot_c_surv_3_groups(lsc17_surv_tbl, lsc17_params,args.OUTPATH, group_weights = Counter(cyt_levels))
    c_index_metrics, c_scores, pca17_surv_tbl, pca17_params, model = cox_models.evaluate(LGN_CDS, LGN_PCA17_params)
    #plot_c_surv_3_groups(pca17_surv_tbl, pca17_params,args.OUTPATH, group_weights = Counter(cyt_levels))
    
    plot_cm(lsc17_surv_tbl, cyt, lsc17_params, args.OUTPATH)
    #plot_cm(pca17_surv_tbl, cyt, pca17_params, args.OUTPATH)
    #plot_scatter()
    plot_cm_any(pca17_surv_tbl, lsc17_surv_tbl, pca17_params, lsc17_params, args.OUTPATH, group_weights =Counter(cyt_levels) )