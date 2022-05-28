from curses.ascii import SUB
from datetime import datetime 
import os 
from engines import utils
from engines.datasets.base_datasets import SurvivalGEDataset
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


def plot_surv_curves(pred_data, HyperParams, surv_curves_outdir, group_weights = [0.5, 0.5] ):
    plt.figure()
    kmf = KaplanMeierFitter()
    median_score = np.median(pred_data["pred_risk"])

    sorted_scores = np.sort(pred_data["pred_risk"])
    nsamples = len(sorted_scores)
    nb_fav = group_weights[0]
    nb_int = group_weights[1]
    int_sep = sorted_scores[nb_fav]
    adv_sep = sorted_scores[nb_fav + nb_int] 
    nc_fav = (pred_data["e"][pred_data["pred_risk"] < int_sep] == 0).sum()
    nc_int = (pred_data["e"][pred_data["pred_risk"] < adv_sep] == 0).sum() - nc_fav
    nc_adv = (pred_data["e"] == 0).sum() - nc_fav - nc_int
    groups = []
    for score in pred_data["pred_risk"]:
        if score < int_sep: groups.append(f"fav. risk (n:{nb_fav} c:{nc_fav})")
        elif score < adv_sep: groups.append(f"int. risk (n:{nb_int} c:{nc_int} )")
        else: groups.append(f"adv. risk (n:{nsamples - nb_fav - nb_int} c:{nc_adv} )")

    pred_data["group"] = groups 
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7,6))
    colors = ["red","blue", "grey"]
    print("plotting ...")
    pdb.set_trace()
    for i, (name, grouped_df) in enumerate( pred_data.groupby("group")):     
        kmf.fit(grouped_df["t"], grouped_df["e"], label=name)
        kmf.plot_survival_function(ax = ax, color = colors[i], show_censors =True, censor_styles = {"ms": 6, "marker": "x"})

    model_type = HyperParams["modeltype"]
    cohort = HyperParams["cohort"]
    model_id = HyperParams["model_id"]
    input_size = HyperParams["input_size"]
    c_ind = HyperParams["c_index_metrics"]
    surv_outpath = os.path.join(surv_curves_outdir, f"{model_type}_{cohort}_{input_size}_{model_id}")
    plt.title(f"Survival curves - model_type: {model_type}")
    ax.set_xlabel(f'''timeline 
    dataset: {cohort}, input_dim: {input_size} 
    c_index: {np.round(c_ind,3)}''')
    ax.grid(visible = True, alpha = 0.5, linestyle = "--")
    plt.tight_layout()
    plt.savefig(f"{surv_outpath}.svg")

def plot_confusion_matrix(pred_scores, cyt, params, outdir):
    # rename groups 
    pred_scores["Cytogenetic risk"] = [{"int":"intermediate cytogenetics", "fav":"favorable cytogenetics", "adv": "adverse cytogenetics"}[g.split(".")[0]] for g in  pred_scores["group"]]
    # merge two files based on index
    scores_merged = cyt.merge(pred_scores, left_index = True, right_index = True)
    # get confusion matrix
    true_cyt = scores_merged["Cytogenetic risk_x"]
    pred_cyt = scores_merged["Cytogenetic risk_y"]
    labels = ["favorable cytogenetics", "intermediate cytogenetics", "adverse cytogenetics"]
    CM = confusion_matrix(true_cyt, pred_cyt, labels = labels)
    # plot
    fig, ax = plt.subplots(figsize = (12,10))
    CM = CM.T
    im = ax.pcolor(CM, vmin = 0, vmax = 177, cmap=plt.cm.Blues)
    for (i, j), z in np.ndenumerate(CM):
        ax.text(j + 0.5, i + 0.5, '{}'.format(int(z)), ha='center', va='center')

    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(CM.shape[1]) + 0.5, minor=False)
    ax.set_xlabel("predicted classes")
    ax.set_ylabel("true classes")
    ax.set_yticks(np.arange(CM.shape[0]) + 0.5, minor=False)
    
    ax.set_xticklabels(labels)
    fig.colorbar(im)
    plt.title(f"Confusion matrix: {params['input_type']}_{params['input_size']}_{params['modeltype']}_{params['cohort']}")
    plt.tight_layout()
    # dump
    plt.savefig(os.path.join(outdir, f"{params['input_type']}_{params['input_size']}_{params['modeltype']}_{params['cohort']}_{params['model_id']}.svg"))



def run(args):
    # get time stamp
    tstamp = "".join(str(datetime.now()).split(".")[:-1]).replace(" ", "_").replace("-", "_").replace(".", "_").replace(":","_")
    # create output dir
    basepath = utils.assert_mkdir(os.path.join("RES", "EXP_4"))
    # survival curves outdir
    surv_curves_outdir = utils.assert_mkdir(os.path.join(basepath, "SURV_CURVES"))
    # confusion matrices outdir
    conf_matrix_outdir = utils.assert_mkdir(os.path.join(basepath, "CONF_MATRIX"))
    # performance tables outdir 
    perf_tables_outdir = utils.assert_mkdir(os.path.join(basepath, "PERF_TABLES"))
    # create runs outdir 
    run_logs_outdir = utils.assert_mkdir(os.path.join(basepath, "RUN_LOGS"))
    # prepare input data
    SGE = SurvivalGEDataset()
    SGE.get_data(args.COHORT)
    CDS = SGE.data["CDS"]
    LSC17 = SGE.data["LSC17"]
    HyperParams = HP_dict(args)
    input_dims = np.arange(args.INPUT_DIMS[0], args.INPUT_DIMS[1], args.INPUT_DIMS[2])
    # prepare performance result matrix
    perfo_matrix = []
    cyt = pd.DataFrame(SGE.data["CF"]["Cytogenetic risk"])
    cyt_levels = [{"intermediate cytogenetics":1, "Intermediate/Normal":1, "adverse cytogenetics": 2, "favorable cytogenetics":0, "Favorable":0, "Standard":1, "Low":0, "Poor":2, None: 1}[level] for level in cyt["Cytogenetic risk"]] 
    cyt["pred_risk"] = cyt_levels
    cyt_c_scores_1, cyt_metrics_1 = functions.compute_cyto_risk_c_index(cyt["pred_risk"],CDS.y, gamma = 0.001, n = args.bootstr_n)
    cyt_c_scores_2, cyt_metrics_2  = functions.compute_aggregated_bootstrapped_c_index(cyt["pred_risk"], CDS.y, n=args.bootstr_n)
    params = HyperParams.generate_default("cytogenetic_risk", cyt["pred_risk"])
    params["c_index_metrics"] = cyt_metrics_1[0]
    params_file_outpath = os.path.join(run_logs_outdir, f"{tstamp}_run_params.csv")            
    params_file = None
    cyt["e"] = CDS.y["e"]
    cyt["t"] = CDS.y["t"]
    print("C index method 1: ", cyt_metrics_1)
    print("C index method 2: ", cyt_metrics_2)
    params["cohort"] = args.COHORT
    plot_c_surv_3_groups(cyt, params, args.OUTPATH, group_weights = Counter(cyt_levels))
    if params_file is None : params_file = pd.DataFrame(params, index = [0])
    else: params_file = pd.concat([params_file, pd.DataFrame(params)])
         
    #print(cyt_metrics)
    perfo_matrix.append(("cyto.risk_scores", cyt_c_scores_1))
    for model_type in args.MODEL_TYPES:  
        # use LSC17 benchmark
        LSC17.split_train_test(HyperParams.nfolds)
        params = HyperParams.generate_default(model_type + "_LSC17", LSC17)
        c_index_metrics, c_scores, surv_tbl, params= cox_models.evaluate(LSC17, params, pca_n = None)
        params["input_type"] = "LSC17"
        params["cohort"] = args.COHORT
        plot_c_surv_3_groups(surv_tbl, params, args.OUTPATH, group_weights = Counter(cyt_levels))
        #plot_confusion_matrix(pred_risks, cyt, params, conf_matrix_outdir)
        params_file = pd.concat([params_file, pd.DataFrame(params, index = [0])])
        perfo_matrix.append((params["model_id"], c_scores))
        for input_dim in input_dims:
            # split training / valid
            data = CDS.clone()
            data.split_train_test(HyperParams.nfolds)
            # generate Hyper-Parameter dict specific to model type
            params = HyperParams.generate_default(model_type + "_PCA", data)
            params["input_type"] = "PCA"
            params["cohort"] = args.COHORT
            c_index_metrics, c_scores, surv_tbl, params= cox_models.evaluate(data, params, pca_n = input_dim) 
            plot_c_surv_3_groups(surv_tbl, params, args.OUTPATH, group_weights = Counter(cyt_levels))
            #plot_confusion_matrix(pred_risks, cyt, params, conf_matrix_outdir)
            params_file = pd.concat([params_file, pd.DataFrame(params, index = [0])])
            perfo_matrix.append((params["model_id"], c_scores))
             # dump params matrix to run logs
            
            params_file.to_csv(params_file_outpath, index=False)
    # dump performance matrix to perf_tables
    perfo_file_outpath = os.path.join(perf_tables_outdir, f"{tstamp}_bootstrapped_c_indices.csv")
    perfo_file = pd.DataFrame(dict(perfo_matrix))
    perfo_file.to_csv(perfo_file_outpath, index = False)

    # output runs logs 
    # run models 
    # print survival curves
    # dump results to table 
    