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
    HyperParams = HP_dict(args)
    data1 = SGE.new(args.COHORT, np.concatenate([mutations, cytogenetics, age_sex]), gene_expressions=None)
    # drop some low variance cols
    data1.x = data1.x[data1.x.columns[np.where(data1.x.var(0) > 0.01)]]
    data1.split_train_test(HyperParams.nfolds)
    params = HyperParams.generate_default("ridge_cph_lifelines_CF", data1)
    c_index_metrics, c_scores, surv_tbl, params= cox_models.evaluate(data1, params, pca_n = None)
    pdb.set_trace()
    # get chrom. rearragments
    # get mutation profile 
    # get binarized age value (>60, <60)
    # combine to LSC17 profile
    # combine to PCA profile 
    
    