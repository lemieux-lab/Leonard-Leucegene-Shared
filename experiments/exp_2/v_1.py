import pandas as pd
from sklearn.metrics import roc_auc_score 
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
from experiments import plotting_functions as plt_f

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
    pca.generate_PCA(300)
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
    pca.generate_PCA(300)
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
    acc_matrix = []
    auc_matrix = []
    for proj_name, ge_data, in_D in ge_data_list:
        for n in tqdm(range(10), desc = proj_name):
            ge_data.split_train_test(5)
            acc_row = [n+1, proj_name, in_D]
            auc_row = [n+1, proj_name, in_D]
            
            for feature in features.keys():
                target = features[feature]    
                acc = []
                scores = []
                targets = []
                for fold in range(5):
                    # train
                    tr_x = ge_data.folds[fold].train.x.iloc[:,:in_D]
                    tr_y = target.loc[tr_x.index]
                    clf = LogisticRegression(penalty = "l2", C = 0.99, max_iter=1000)
                    clf.fit(tr_x, tr_y)
                    # test 
                    tst_x = ge_data.folds[fold].test.x.iloc[:,:in_D]
                    tst_y = target.loc[tst_x.index]
                    acc.append(clf.score(tst_x, tst_y))  
                    scores = np.concatenate([scores, clf.predict_proba(tst_x)[:,1].flatten()])
                    targets = np.concatenate([targets, tst_y])
                auc = roc_auc_score(targets, scores)
                acc_row.append(np.mean(acc))
                auc_row.append(auc)
            acc_matrix.append(acc_row)
            auc_matrix.append(auc_row)
    acc = pd.DataFrame(acc_matrix, columns = header)
    auc = pd.DataFrame(auc_matrix, columns = header)
    return acc, auc

def perform_projection(proj_type, data, input_size = 17):    
    # set data
    if proj_type == "PCA":
        data = data["CDS"].clone()
        data.generate_PCA(input_size)
    elif proj_type == "SVD":
        data = data["CDS"].clone()
        data.generate_SVD(input_size)
    elif proj_type == "RPgauss":
        data = data["CDS"].clone()
        data.generate_RP("gauss", input_size)
    elif proj_type == "RPgauss_var":
        data = data["CDS"].clone()
        data.generate_RP("gauss", input_size, var_frac = 0.5)
    elif proj_type == "RPsparse":
        data = data["CDS"].clone()
        data.generate_RP("sparse", input_size)
    elif proj_type == "RSelect":
        data = data["CDS"].clone()
        data.generate_RS(input_size, var_frac = 0.5)
    elif proj_type == "CF-PCA":
        y = data['CDS'].y
        gi = data["CDS"].gene_info
        red_bl_cf = data["CF_bin"].iloc[:,[0,1,2,3,4,5,7,9,11]]
        pca = data["CDS"].clone()
        pca.generate_PCA(input_size)
        red_bl_cf_pca = red_bl_cf.merge(pca.x.iloc[:,:input_size], left_index = True, right_index = True)
        data = Data(red_bl_cf_pca, y, gi , name = f"red_bl_PCA_d_{input_size}" )
    
    elif proj_type == "CF-LSC17":
        y = data['LSC17'].y
        gi = data["LSC17"].gene_info
        red_bl_cf = data["CF_bin"].iloc[:,[0,1,2,3,4,5,7,9,11]]
        lsc17 = data["LSC17"].clone()
        red_bl_cf_lsc17 = red_bl_cf.merge(lsc17.x, left_index = True, right_index = True)
        data = Data(red_bl_cf_lsc17, y, gi , name = f"red_bl_LSC17" )
    
    elif proj_type == "CF":
        y = data['CDS'].y
        gi = data["CDS"].gene_info
        return Data(data["CF_bin"].iloc[:,[0,1,2,3,4,5,7,9,11]], y, gi, name = "Clinical features (binarized)")
    elif proj_type == "CYT":
        y = data['CDS'].y
        gi = data["CDS"].gene_info
        return Data(data["CF_bin"].iloc[:,[1,2,3]], y , gi, name = "Cytogenetic risk baseline")
    elif proj_type == "LSC17":
        data = data[proj_type].clone()
        data.name = "LSC17"
    return data 
    
def get_proj_type_cph_scores_perf(cohort_data, cohort, proj_types, fig_outdir, bootstrap_n = 1000):
    nsamples = cohort_data["CDS"].x.shape[0]
    nfolds = nsamples # leave one out 
    # create datasets by proj_type
    ge_cf_list = []
    for p in proj_types:
        if "PCA" in p:
            input_size = int(p[-2:])
            p = p[:-2]
        else : input_size = 17
        ge_cf_list.append(perform_projection(p, cohort_data, input_size = input_size))
        
    # init empty ret matrix
    scores_matrix = []
    perfo_matrix = []
    # cycle through datasets
    for data in ge_cf_list:
        # shuffle dataset!
        # idx = np.arange(data.x.shape[0])
        # np.random.shuffle(idx) # shuffle dataset! 
        # data.reindex(idx)
        # init empty scores 
        tst_scores = [] # store risk prediction scores for agg_c_index calc
        # split data
        data.split_train_test(nfolds, loo = True)
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
        scores_matrix.append(risk_scores)
        perfo_list = []
        for i in tqdm(range(bootstrap_n), desc = "bootstraping c_index"):
            idx = np.random.choice(np.arange(nsamples), nsamples, replace = True)
            c_ind_tst = functions.compute_c_index(data.y["t"][idx], data.y["e"][idx], risk_scores[idx])
            perfo_list.append(c_ind_tst)
        perfo_matrix.append(perfo_list)
        plt_f.plot_hi_risk_lo_risk(risk_scores, pd.DataFrame(data.y), fig_outdir, data.name, cohort, perfo_list)
        #cox_nll = functions.compute_cox_nll(data.y["t"], data.y["e"], risk_scores)
    #scores = pd.DataFrame(np.array(scores_matrix).T, columns = proj_types, index = ge_cf_list[0].x.index)
    #perfo = pd.DataFrame(np.array(perfo_matrix).T, columns = proj_types)
    #return scores, perfo 



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

def heatmap(tbl, columns, fig_path, proj_type, cohort, figsize = (10,10)):
    fig_outfile =os.path.join(fig_path, f"corr_cf_{proj_type}_{cohort}_heatmap.svg" )
    mini_fig_outfile = os.path.join(fig_path, f"corr_cf_{proj_type}_{cohort}_heatmap_mini.svg" )
        
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
    plt.savefig(fig_outfile)
    dat = dat.iloc[::-1,:]
    fig, ax = plt.subplots(figsize = figsize)
    im = ax.imshow(abs(dat), vmin = 0, vmax = 1, cmap=plt.cm.Blues)
    ax.set_yticklabels(dat.index)
    ax.set_xticks(np.arange(dat.shape[1]), minor=False)
    ax.set_yticks(np.arange(dat.shape[0]), minor=False)
    ax.xaxis.tick_top()
    ax.set_xticklabels(dat.columns, rotation = 45)
    fig.colorbar(im)
    plt.savefig(mini_fig_outfile)
def log_reg_heatmap(tbl, columns, figsize = (10,10)):
      
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

def action4(cohort_data, cohort, fig_path):
    # PCA17, PCA300, LSC17 Logistic Regression on CF
    print(f"Action 4: Train Logistic Regression on LSC17, PCA17, PCA300, to predict CF. By 5-fold cross-val, 10 replicates.")
    acc_outfile = os.path.join(fig_path, f"log_reg_lsc17_pca_to_cf_ACC_{cohort}.csv" )   
    acc_fig_outfile = os.path.join(fig_path, f"log_reg_lsc17_pca_to_cf_ACC_{cohort}.svg" )     
    auc_outfile = os.path.join(fig_path, f"log_reg_lsc17_pca_to_cf_AUC_{cohort}.csv" )   
    auc_fig_outfile = os.path.join(fig_path, f"log_reg_lsc17_pca_to_cf_AUC_{cohort}.svg" )     
     
    log_reg_res_acc, log_reg_res_auc = get_pca_lsc17_log_reg_perf_to_cf(cohort_data)
    log_reg_res_acc.to_csv(acc_outfile)
    log_reg_res_auc.to_csv(auc_outfile)
    # add heatmap of results --> heatmap_log_reg.svg
    # preprocess 
    acc = log_reg_res_acc.groupby("proj_type").mean().reset_index()
    auc = log_reg_res_auc.groupby("proj_type").mean().reset_index()
    acc.index = acc.proj_type
    acc = acc.iloc[:, np.arange(3,10)]
    auc.index = auc.proj_type
    auc = auc.iloc[:, np.arange(3,10)]
    
    ax = log_reg_heatmap(acc, columns = ["adv_risk_y", "fav_risk_y", 
    "int_risk_y", "npm1", "idh1", "flt3", "sex"], figsize = (9,3))
    plt.savefig(acc_fig_outfile)

    ax = log_reg_heatmap(auc, columns = ["adv_risk_y", "fav_risk_y", 
    "int_risk_y", "npm1", "idh1", "flt3", "sex"], figsize = (9,3))
    plt.savefig(auc_fig_outfile)
    print("done")

def action5(cohort_data, cohort, proj_types, outpath, bootstrap_n = 1000, plot_cyto = False):
    # CF (reduced bl) + LSC17 / PCA --> survival (with CPH)
        # add a multi variate CPH trained on all features but cyto risk
        # add CPH on cyto risk only
        # add bootstraped c_index (1000)
        # output scores by samples, with CF
        # foreach input feature type (PCA-CF, PCA, CF, LSC17, LSC17-CF, CR, CR'):
        #   output the inferred CPH covariate beta factors distributions, p-values??
        #   output performance metric (could we also get Loss? Brier-Score?)
    if 1: 
        print("Action 5: Getting the performance of CPH with input features with survival curves:")
        print("A) reduced CF baseline")
        print("B) reduced CF baseline + LSC17")
        print("C) reduced CF baseline + PCA17")
        print("D) PCA17 only")
        print("E) LSC17 only")
        print("F) Plot CYT only on cohort")
    fig_outdir = utils.assert_mkdir(outpath) 
    #scores_outfile = os.path.join(outdir, f"{cohort}_scores_by_method.csv" )
    #perfo_outfile = os.path.join(outdir, f"{cohort}_bootstrap_{bootstrap_n}_by_method.csv" ) 
    # cytogenetic risk only features
    get_proj_type_cph_scores_perf(cohort_data, cohort, proj_types, fig_outdir, bootstrap_n = bootstrap_n)
    # merge with CF file
    # dump
    if plot_cyto:
        CYT  = perform_projection("CYT", cohort_data)
        plt_f.plot_CYT_km_curves(CYT.x, CYT.y, cohort, fig_outdir)
    # scores = scores.merge(cohort_data["CDS"].y, right_index = True, left_index = True)
    # scores = scores.merge(CYT.x, right_index = True, left_index = True)
    
    print("done")

def action1(args):
    print("Action 1: Loading data, preprocessing data...")
    SGE = SurvivalGEDataset()
    cohort_data = SGE.get_data(args.COHORT)
    print("done")
    return cohort_data

def action2(cohort_data, cohort, fig_path):
    outfile = os.path.join(fig_path, f"summary_table_cf_{cohort}.csv" )
    fig_outfile =  os.path.join(fig_path, f"summary_table_cf_{cohort}.svg")
    print(f"Action 2: Printing summary data table of clinical infos. --> {outfile, fig_outfile}")
    table = get_summary_table_cf(cohort_data)
    table.to_csv(outfile)
    axes = barplots(table)
    axes.set_ylim([0,1])
    plt.savefig(fig_outfile)
    # bar plots of clinical features occ. --> barplot_cf.svg
    print("done") 

def action3(cohort_data, cohort, fig_path):
    # mini version without text (squares)
    outfile =os.path.join(fig_path, f"corr_pca_cf_{cohort}.csv" )   
    fig2_outfile =os.path.join(fig_path, f"pca_expl_var_{cohort}.svg" )
    print(f"Action 3: Corr PCA to clinical features to CF. {outfile}")
    res = get_pca_lsc17_corr_to_cf(cohort_data)
    for proj_type, corr_df in res.items():
        corr_df.to_csv(outfile)
        # plot heatmap of table --> heatmap_pca.svg
        heatmap(corr_df, ['Age_at_diagnosis', 'adverse cytogenetics', 
        'favorable cytogenetics', 'intermediate cytogenetics',
        'NPM1 mutation_1.0', 'IDH1-R132 mutation_1.0', 
        'FLT3-ITD mutation_1', 'Sex_F'],fig_path, proj_type, cohort, figsize = (12,12))
        
        # plot expl. var / var ratio vs #PC --> pca var .svg
        if proj_type == "pca": 
            ax = pc_var_plot(corr_df)
            plt.savefig(fig2_outfile)
    print("done")

def run(args):
    utils.assert_mkdir(args.OUTPATH)
    # load data, set up basepath
    cohort_data = action1(args)

    # plots the features frequency
    if 0: action2(cohort_data, args.COHORT, args.OUTPATH)
    
    # plots the CORR to PCs with heatmap (full, mini) + var_expl plot
    if 0: action3(cohort_data, args.COHORT, args.OUTPATH)
    
    if "LOG_REG" in args.MODEL_TYPES:
        # perform logistic regression fitting
        # add AUC heatmap please
        action4(cohort_data, args.COHORT, args.OUTPATH)

    if "CPH" in args.MODEL_TYPES:
        # perform cph benchmark computations
        
        action5(cohort_data, args.COHORT, args.PROJ_TYPES, args.OUTPATH, bootstrap_n = args.NREP_TECHN, plot_cyto = args.PLOT_CYTO_RISK)



