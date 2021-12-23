import pandas as pd
from sklearn import base
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
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

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
    data = cohort_data["CF_bin"]
    stats = pd.DataFrame(data.min(0), columns = ["min"]).merge(
            pd.DataFrame(data.max(0), columns = ["max"]), right_index = True, left_index=True).merge(
            pd.DataFrame(data.mean(0), columns = ["mean"]), right_index = True, left_index=True).merge(
            pd.DataFrame(data.median(0), columns = ["median"]), right_index = True, left_index=True)
    return stats

def get_pca_lsc17_corr_to_cf(cohort_data):
    # clinical binary
    bin_cf = cohort_data["CF_bin"]
    lsc17 = cohort_data["LSC17"].clone()
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
    pca_corr = pd.DataFrame(matrix, columns = header)
    matrix = []
    for i in range(17):
        row = [lsc17.x.columns[i]]
        for feature in bin_cf.columns:
            row.append(np.corrcoef(lsc17.x.iloc[:,i], bin_cf[feature])[0,1])
        matrix.append(row)
    lsc17_corr = pd.DataFrame(matrix, columns = np.concatenate([["gene_name"], bin_cf.columns]))
    # get corr to features with Lsc17
    return {"pca": pca_corr, "lsc17":lsc17_corr}

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

def get_pca_lsc17_cf_cph_perf(cohort_data, cohort, basepath):
    nfolds = 300
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
    header = ["proj_type","c_ind_tst", "cox_nll"]
    matrix = []
    # cycle through datasets
    for data in ge_cf_list:
        # shuffle dataset!
        idx = np.arange(data.x.shape[0])
        np.random.shuffle(idx) # shuffle dataset! 
        data.reindex(idx)
        losses = []
        # init empty scores 
        tst_scores = [] # store risk prediction scores for agg_c_index calc
        # split data
        data.split_train_test(nfolds)
        # cycle through folds
        for foldn in tqdm(range(nfolds), desc = f"leave-one-out CPH-{data.name}"):
            test_data = data.folds[foldn].test
            train_data = data.folds[foldn].train
            # choose model type, hps and train
            model = cox_models.CPH(data = train_data)
            model.set_fixed_params({"input_size": train_data.x.shape[1], "wd": 1e-10})
            tr_metrics = model._train()
            # test
            tst_metrics = model._test(test_data, c_index = False)
            tst_scores.append(tst_metrics["out"])
        risk_scores = np.concatenate(tst_scores)
        c_ind_tst = functions.compute_c_index(data.y["t"], data.y["e"], risk_scores)
        cox_nll = functions.compute_cox_nll(data.y["t"], data.y["e"], risk_scores)

        # add brier score, loss
        # split hi-risk, lo-risk
        plot_hi_risk_lo_risk(risk_scores, pd.DataFrame(data.y), basepath, data.name, cohort, c_ind_tst)
        matrix.append([data.name, c_ind_tst, cox_nll])
    ret_df = pd.DataFrame(matrix, columns = header)
    
    print(ret_df)
    return ret_df 

def plot_hi_risk_lo_risk(risk_scores, data, basepath, proj_type, cohort, c):
    plt.figure()
    # split hi-risk lo-risk
    kmf = KaplanMeierFitter() 
    # hi_risk
    hi_risk = risk_scores > np.median(risk_scores)
    # low_risk
    lo_risk = risk_scores < np.median(risk_scores)
    kmf.fit(data[hi_risk].t, data[hi_risk].e, label = "Hi risk")
    a1 = kmf.plot()
    kmf.fit(data[lo_risk].t, data[lo_risk].e, label = "Low risk") 
    kmf.plot(ax=a1)
    # log rank test
    
    results=logrank_test(data[hi_risk].t,data[lo_risk].t,event_observed_A= data[hi_risk].e, event_observed_B=data[hi_risk].e)
    a1.set_title(f"Predicted KM survival curves High risk vs low {proj_type} {cohort} c = {c}\n log-rank test p_val: {results._p_value}")
    plt.savefig(os.path.join(basepath, f"{proj_type}_cph_{cohort}_kaplan_meier.svg"))



def boxplots(benchmark):
    plt.figure()
    axes = benchmark[["c_ind_tst", "proj_type"]].boxplot(by="proj_type", grid = False, showfliers = False, return_type = 'axes')
    axes[0].set_title(f"Performance of CPH by input type with Leucegene")
    axes[0].set_xlabel("Input type")
    axes[0].set_ylabel("Performance (concordance index)")
    axes[0].set_ylim([0.5, 0.75])
    return axes

def barplots(tbl):
    plt.figure()
    features_to_plot = ["adverse cytogenetics", "favorable cytogenetics", "intermediate cytogenetics", "NPM1 mutation_1.0", "IDH1-R132 mutation_1.0", "FLT3-ITD mutation_1", "Sex_F", "Sex_M"]
    dat = tbl.loc[features_to_plot]["mean"]
    dat.index = ["adv cyt", "fav cyt", "int cyt", "NPM1", "IDH1", "FLT3-ITD", "Females", "Males"]
    ax = dat.plot.bar(rot = 45)
    return ax 

def heatmap(tbl, columns, figsize = (10,10)):
    # filter columns
    dat = tbl[columns].iloc[::-1]
    fig, ax = plt.subplots(figsize = figsize)
    im = ax.pcolor(abs(dat), vmin = 0, vmax = 1, cmap=plt.cm.Blues)
    for (i, j), z in np.ndenumerate(dat):
        ax.text(j + 0.5, i + 0.5, '{:0.2f}'.format(z), ha='center', va='center')

    ax.set_yticklabels(dat.index)
    ax.set_xticks(np.arange(dat.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(dat.shape[0]) + 0.5, minor=False)
    ax.xaxis.tick_top()
    ax.set_xticklabels(dat.columns, rotation = 45)
    fig.colorbar(im)
    return ax

def pc_var_plot(tbl):
    fig, ax = plt.subplots(figsize = (10,2))
    ax.bar(x = tbl.index, height = np.log10(tbl.expl_var))
    ax.plot(np.arange(tbl.shape[0]), np.log10(np.cumsum(tbl.expl_var)))
    ax.set_ylim([-1,np.log10(100)])
    ax.set_xticklabels(np.arange(tbl.shape[0]))
    ax.set_xlabel("Principal component #")
    ax.set_ylabel("log(% Explained variance)")

def action4(basepath, cohort_data, cohort):
    # PCA17, PCA300, LSC17 Logistic Regression on CF
    print(f"Action 4: Train Logistic Regression on LSC17, PCA17, PCA300, to predict CF. By 5-fold cross-val, 10 replicates.")
    outfile = os.path.join(basepath, f"log_reg_lsc17_pca_to_cf_{cohort}.csv" )   
    fig_outfile = os.path.join(basepath, f"log_reg_lsc17_pca_to_cf_{cohort}.svg" )     
    log_reg_res = get_pca_lsc17_log_reg_perf_to_cf(cohort_data)
    log_reg_res.to_csv(outfile)
    # add heatmap of results --> heatmap_log_reg.svg
    # preprocess 
    dat = log_reg_res.groupby("proj_type").mean().reset_index()
    dat.index = dat.proj_type
    dat = dat.iloc[:, np.arange(3,10)]
    ax = heatmap(dat, columns = ["adv_risk_y", "fav_risk_y", 
    "int_risk_y", "npm1", "idh1", "flt3", "sex"], figsize = (9,3))
    plt.savefig(fig_outfile)
    print("done")

def action5(cohort_data, cohort, basepath):
    # CF (reduced bl) + LSC17 / PCA --> survival (with CPH)
        # add a multi variate CPH trained on all features but cyto risk
        # add CPH on cyto risk only
        # output survivor curves by (true) cyto risk
        # foreach input feature type (PCA-CF, PCA, CF, LSC17, LSC17-CF, CR, CR'):
        #   output the inferred CPH covariate beta factors distributions, p-values??
        #   output performance metric (could we also get Loss? Brier-Score?)
    print("Action 5: Getting the performance of A) reduced CF baseline with CoxPH. B) reduced CF baseline + LSC17 C) reduced CF baseline + PCA17 ")
    table_outfile = os.path.join(basepath, f"benchmark_bl_cf_lsc17_pca_cph_{cohort}.csv" ) 
    benchmark = get_pca_lsc17_cf_cph_perf(cohort_data, cohort, basepath)
    benchmark.to_csv(table_outfile)
    print("done")
    
def run(args):
    
    for cohort in args.COHORTS:
        print("Action 1: Loading data, preprocessing data...")
        SGE = SurvivalGEDataset()
        cohort_data = SGE.get_data(cohort)
        basepath = utils.assert_mkdir(os.path.join("RES", f"EXP_{args.EXP}"))
        print("done")

        outfile = os.path.join(basepath, f"summary_table_cf_{cohort}.csv" )
        fig_outfile =  os.path.join(basepath, f"summary_table_cf_{cohort}.svg")
        print(f"Action 2: Printing summary data table of clinical infos. --> {outfile, fig_outfile}")
        table = get_summary_table_cf(cohort_data)
        table.to_csv(outfile)
        axes = barplots(table)
        axes.set_ylim([0,1])
        plt.savefig(fig_outfile)
        # bar plots of clinical features occ. --> barplot_cf.svg
        print("done") 

        print(f"Action 3: Corr PCA to clinical features to CF. {outfile}")
        outfile =os.path.join(basepath, f"corr_pca_cf_{cohort}.csv" )   
        fig2_outfile =os.path.join(basepath, f"pca_expl_var_{cohort}.svg" )
        
        res = get_pca_lsc17_corr_to_cf(cohort_data)
        for proj_type, corr_df in res.items():
            fig_outfile =os.path.join(basepath, f"corr_cf_{proj_type}_{cohort}_heatmap.svg" )
            corr_df.to_csv(outfile)
            # plot heatmap of table --> heatmap_pca.svg
            ax = heatmap(corr_df, ['Age_at_diagnosis', 'adverse cytogenetics', 
            'favorable cytogenetics', 'intermediate cytogenetics',
            'NPM1 mutation_1.0', 'IDH1-R132 mutation_1.0', 
            'FLT3-ITD mutation_1', 'Sex_F'], figsize = (12,12))
            plt.savefig(fig_outfile)
            # plot expl. var / var ratio vs #PC --> pca var .svg
            if proj_type == "pca": 
                ax = pc_var_plot(corr_df)
                plt.savefig(fig2_outfile)
        print("done")
        
        # perform logistic regression fitting
        if 1: action4(basepath, cohort_data, cohort)
        
        # perform cph benchmark computations
        if 0: action5(cohort_data, cohort, basepath)
        



