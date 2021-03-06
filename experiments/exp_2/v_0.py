import pandas as pd
from engines.base_engines import Benchmark
from engines.datasets.base_datasets import SurvivalGEDataset
from engines import utils
import os
import matplotlib.pyplot as plt
import pdb
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from engines.datasets.base_datasets import Data
from engines.models import cox_models
import engines.models.functions as functions 

def get_summary_table_cf(cohort_data):
    """
    returns a list of interesting dataframes carrying stats and infos
    """
    ret_list = []
    # group by cell differentiation WHO2008 + FAB
    # get number by group
    who_count = cohort_data["CF"].groupby("WHO classification").count()
    fab_count = cohort_data["CF"].groupby("FAB classification").count() 
    # group by Cytogenticity
    cg_group = cohort_data["CF"].groupby("Cytogenetic group").count()
    cg_risk = cohort_data["CF"].groupby("Cytogenetic risk").count()
    # group by therapy
    # binarize
    # dump 
    d = cohort_data["CF_bin"]
    stats = pd.DataFrame(d.min(0), columns = ["min"]).merge(
            pd.DataFrame(d.max(0), columns = ["max"]), right_index = True, left_index=True).merge(
            pd.DataFrame(d.mean(0), columns = ["mean"]), right_index = True, left_index=True).merge(
            pd.DataFrame(d.median(0), columns = ["median"]), right_index = True, left_index=True)
    return stats

def get_pca_lsc17_corr_to_cf(cohort_data):
    # clinical binary
    bin_cf = cohort_data["CF_bin"]
    # generate PCA
    pca = cohort_data["CDS"].clone()
    pca.generate_PCA()
    var = pca._pca.explained_variance_ / pca._pca.explained_variance_.sum() * 100
    header = np.concatenate((["PC", "expl_var"], bin_cf.columns))
    matrix = []
    for i in range(30):
        row = [f"PC{i+1}", var[i]]
        for feature in bin_cf.columns:
            row.append(np.corrcoef(pca.x[i],bin_cf[feature])[0,1])
        matrix.append(row)
        # compute corr to feature
    ret_df = pd.DataFrame(matrix, columns = header)
    return ret_df
    # get corr to features 

def get_pca_lsc17_log_reg_perf_to_cf(cohort_data):
    pca = cohort_data["CDS"].clone()
    pca.generate_PCA()
    ge_data_list = [
        ("lsc17", cohort_data["LSC17"], 17),
        ("pca17", pca, 17),
        ("pca300", pca, 300)
    ]
    y = cohort_data["CF_bin"]
    features = {"adv_risk_y": y.iloc[:,2],
    "fav_risk_y": y.iloc[:,3],
    "int_risk_y": y.iloc[:,4],
    "npm1": y.iloc[:,5],
    "idh1": y.iloc[:,7],
    "flt3": y.iloc[:,9],
    "sex": y.iloc[:,11]}
    header = np.concatenate([["repn","proj_type", "nb_input_f"], list(features.keys())] )
    matrix = []
    for proj_name, ge_data, in_D in ge_data_list:
        for n in tqdm(range(10), desc = proj_name):
            ge_data.split_train_test(5)
            row = [n+1, proj_name, in_D]
            for feature in features.keys():
                target = features[feature]    
                scores = []
                for fold in range(5):
                    # train
                    tr_x = ge_data.folds[fold].train.x.iloc[:,:in_D]
                    tr_y = target.loc[tr_x.index]
                    clf = LogisticRegression(penalty = "l2", C = 0.99, max_iter=1000)
                    clf.fit(tr_x, tr_y)
                    # test 
                    tst_x = ge_data.folds[fold].test.x.iloc[:,:in_D]
                    tst_y = target.loc[tst_x.index]
                    scores.append(clf.score(tst_x, tst_y))  
                row.append(np.mean(scores))
            matrix.append(row)
    ret_df = pd.DataFrame(matrix, columns = header)
    return ret_df

def get_pca_lsc17_cf_cph_perf(cohort_data):
    y = cohort_data['CDS'].y
    gi = cohort_data["CDS"].gene_info
    # get reduced baseline , get cph perfo
    red_bl_cf = cohort_data["CF_bin"].iloc[:,[0,1,2,3,4,5,7,9,11]]
    # add PCA info to reduced baseline, get cph perfo
    pca = cohort_data["CDS"].clone()
    pca.generate_PCA()
    red_bl_cf_pca17 = red_bl_cf.merge(pca.x.iloc[:,:17], left_index = True, right_index = True)
    # add LSC17 info to reduced baseline, get cph perfo
    lsc17 = cohort_data["LSC17"].clone()
    red_bl_cf_lsc17 = red_bl_cf.merge(lsc17.x, left_index = True, right_index = True)
    ge_cf_list = [
        Data(red_bl_cf, y, gi, name = "reduced_baseline"), 
        Data(red_bl_cf_pca17, y, gi , name = "red_bl_PCA17" ),
        Data(red_bl_cf_lsc17, y, gi , name = "red_bl_cf_LSC17"),
        Data(pca.x.iloc[:,:17], y, gi, name = "PCA17"),
        Data(lsc17.x, y, gi, name = "LSC17"),
    ]
    # init empty ret matrix
    header = ["proj_type","rep_n", "c_ind_tst"]
    matrix = []
    # cycle through datasets
    for data in ge_cf_list:
        for repn in tqdm(range (10), desc = data.name):
            # shuffle dataset!
            idx = np.arange(data.x.shape[0])
            np.random.shuffle(idx) # shuffle dataset! 
            data.reindex(idx)
            # init empty row
            row = [data.name, repn]
            # init empty scores 
            tst_scores = [] # store risk prediction scores for agg_c_index calc
            # split data
            data.split_train_test(5)
            # cycle through folds
            for foldn in range(5):
                test_data = data.folds[foldn].test
                train_data = data.folds[foldn].train
                # choose model type, hps and train
                model = cox_models.CPH(data = train_data)
                model.set_fixed_params({"input_size": train_data.x.shape[1], "wd": 1e-10})
                tr_metrics = model._train()
                # test
                tst_metrics = model._test(test_data)
                tst_scores.append(tst_metrics["out"])
            c_ind_tst = functions.compute_c_index(data.y["t"], data.y["e"], np.concatenate(tst_scores))
            row.append(c_ind_tst)
            matrix.append(row)
    ret_df = pd.DataFrame(matrix, columns = header)
    return ret_df 

def boxplots(benchmark):
    plt.figure()
    axes = benchmark[["c_ind_tst", "proj_type"]].boxplot(by="proj_type", grid = False, showfliers = False, return_type = 'axes')
    axes[0].set_title(f"Performance of CPH by input type with Leucegene")
    axes[0].set_xlabel("Input type")
    axes[0].set_ylabel("Performance (concordance index)")
    axes[0].set_ylim([0.5, 0.75])
    return axes

def run(args):
    
    for cohort in args.COHORTS:
        print("Action 1: Loading data, preprocessing data...")
        SGE = SurvivalGEDataset()
        cohort_data = SGE.get_data(cohort)
        basepath = utils.assert_mkdir(os.path.join("RES", f"EXP_{args.EXP}"))
        print("done")

        outfile = os.path.join(basepath, f"summary_table_cf_{cohort}.csv" )
        print(f"Action 2: Printing summary data table of clinical infos. --> {outfile}")
        get_summary_table_cf(cohort_data).to_csv(outfile)
        print("done") 

        print(f"Action 3: Corr PCA to clinical features to CF. {outfile}")
        outfile =os.path.join(basepath, f"corr_pca_cf_{cohort}.csv" )   
        get_pca_lsc17_corr_to_cf(cohort_data).to_csv(outfile)
        print("done")

        # PCA17, PCA300, LSC17 Logistic Regression on CF
        #print(f"Action 4: Train Logistic Regression on LSC17, PCA17, PCA300, to predict CF. By 5-fold cross-val, 10 replicates.")
        #outfile = os.path.join(basepath, f"log_reg_lsc17_pca_to_cf_{cohort}.csv" )   
        #get_pca_lsc17_log_reg_perf_to_cf(cohort_data).to_csv(outfile)

        
        # CF (reduced bl) + LSC17 / PCA --> survival (with CPH)
        print("Action 5: Getting the performance of A) reduced CF baseline with CoxPH. B) reduced CF baseline + LSC17 C) reduced CF baseline + PCA17 ")
        table_outfile = os.path.join(basepath, f"benchmark_bl_cf_lsc17_pca_cph_{cohort}.csv" ) 
        fig_outfile = os.path.join(basepath, f"benchmark_bl_cf_lsc17_pca_cph_{cohort}.svg" )  
        benchmark = get_pca_lsc17_cf_cph_perf(cohort_data)
        benchmark.to_csv(table_outfile)
        axes = boxplots(benchmark)
        plt.savefig(fig_outfile)
        print("done")
        
        pdb.set_trace()
        # 60'
        print(f"Action 6: Kaplan-Meier Curve of our survival data. --> {outfile}")
        outfile =os.path.join(basepath, f"kaplan_meier_{cohort}.csv" ) 
        ax = data.plot_kaplan_meier_curve()
        plt.savefig(outfile)
        print("done")

        

        # 60''
        # Gene Exp + single clinical feature type at a time
        
        # 60''
        # Gene Exp + all cf features - one type

        # 30'        
        # All cf Features + Gene Exp with increasing input dim

        # 120'' Figure Prep + # 60' Writing
        # 30' Prep 
        # 180' Cleanup + Pres 
        
        # total: 12h


