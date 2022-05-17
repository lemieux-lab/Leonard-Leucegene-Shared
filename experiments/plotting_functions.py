import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none" # text rendering in figures output 
import pdb 
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import os

def plot_tsne(tsne_data, groups_df, groups, basepath):
    tsne = None 
    # plot tsne
    return tsne

def plot_overfit(data, basepath):
    fig, ax = plt.subplots()
    return ax

def plot_c_surv(cox_output, surv_curves_outdir, group_weights = [0.5, 0.5] ):
    c_scores = cox_output[1]
    pred_data = cox_output[2]
    HyperParams = cox_output[3]
    plt.figure()
    kmf = KaplanMeierFitter()
    median_score = np.median(pred_data["pred_risk"])

    sorted_scores = np.sort(pred_data["pred_risk"])
    nsamples = len(sorted_scores)
    #groups = [["high risk", "low risk"][int(sample >= median_score)] for sample in pred_data["pred_risk"]]

    nb_hi = int(float(nsamples / 2))
    nc_hi = (pred_data["e"][pred_data["pred_risk"] >= median_score] == 0).sum()
    nc_lo = (pred_data["e"][pred_data["pred_risk"] < median_score] == 0).sum()
    groups = []
    for score in pred_data["pred_risk"]:
        if score >= median_score: groups.append(f"high. risk (n:{nb_hi} c:{nc_hi})")
        else: groups.append(f"low risk. risk (n:{nsamples - nb_hi} c:{nc_lo} )")

    pred_data["group"] = groups 
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (7,6))
    colors = ["red","blue", "grey"]
    print("plotting ...")
    for i, (name, grouped_df) in enumerate( pred_data.groupby("group")):     
        kmf.fit(grouped_df["t"], grouped_df["e"], label=name)
        kmf.plot_survival_function(ax = ax, color = colors[i], show_censors =True, censor_styles = {"ms": 6, "marker": "x"})

    model_type = HyperParams["modeltype"]
    cohort = HyperParams["cohort"]
    model_id = HyperParams["model_id"]
    input_size = HyperParams["input_size"]
    c_ind = HyperParams["c_index_metrics"]
    input_type = HyperParams["input_type"]
    surv_outpath = os.path.join(surv_curves_outdir, f"5_{input_type}_{model_type}_{cohort}_{input_size}")
    plt.title(f"Survival curves - model_type: {model_type}")
    ax.set_xlabel(f'''timeline 
    dataset: {cohort}, Input type: {input_type}
    input_dim: {input_size} c_index: {np.round(c_ind,3)}''')
    ax.grid(visible = True, alpha = 0.5, linestyle = "--")
    plt.tight_layout()
    plt.savefig(f"{surv_outpath}.svg")

def plot_correlations(data1,data2, basepath):
    fig, ax = plt.subplots()
    return ax, basepath

def plot_variance(pca_data, basepath):
    fig, ax = plt.subplots()
    return ax, basepath

def plot_c_surv_3_groups(cox_output, surv_curves_outdir, group_weights = [0.5, 0.5] ):
    c_scores = cox_output[1]
    pred_data = cox_output[2]
    HyperParams = cox_output[3]
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

def plot_cm(data1, data2, basepath):
    fig, ax = plt.subplots()
    return ax, basepath

def plot_training(loss_training, c_index_training, foldn, model):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('steps')
    ax1.set_ylabel('loss')
    ax1.scatter(np.arange(len(loss_training)), loss_training, label = "loss", color = "blue")
    ax1.plot(np.arange(len(loss_training)), loss_training, label = "loss", color = "blue")
    ax1.tick_params(axis = "y")
    ax2 = ax1.twinx()
    ax2.scatter(np.arange(len(c_index_training)), c_index_training, label = "c_index", color = "orange")
    ax2.plot(np.arange(len(c_index_training)), c_index_training, label = "c_index", color = "orange")
    ax2.tick_params(axis='y')
    ax2.set_ylabel('c_index') 
    plt.title(f"Training loss and c_index thru optimization {model} on Leucegene data")
    fig.tight_layout()
    plt.savefig(f"fig_dump/{model}_fold_{foldn}_tr_loss_c_ind")
    
def plot_hi_risk_lo_risk(risk_scores, data, fig_outdir, proj_type, cohort, wd, perfo_list):
    plt.figure(figsize = (6,5))
    # split hi-risk lo-risk
    kmf = KaplanMeierFitter() 
    # hi_risk
    hi_risk = risk_scores > np.median(risk_scores)
    # low_risk
    lo_risk = risk_scores <= np.median(risk_scores)
    kmf.fit(data[hi_risk].t, data[hi_risk].e, label = "Hi risk")
    a1 = kmf.plot()
    kmf.fit(data[lo_risk].t, data[lo_risk].e, label = "Low risk") 
    kmf.plot(ax=a1)
    # log rank test
    results=logrank_test(data[hi_risk].t,data[lo_risk].t,event_observed_A= data[hi_risk].e, event_observed_B=data[lo_risk].e)
    # get bootstrap results 
    c_ind = np.sort(perfo_list)
    low_c_ind = c_ind[int(0.025 * len(c_ind))]
    hi_c_ind = c_ind[int(0.975 * len(c_ind))]
    a1.set_title(f"Predicted KM survival curves High risk vs low {proj_type} \n {cohort} c = {np.median(c_ind).round(4)}, ({low_c_ind.round(4)}, {hi_c_ind.round(4)})\n log-rank test p_val: {results._p_value}")
    a1.set_ylim([0,1])
    plt.savefig(os.path.join(fig_outdir, f"{cohort}_{proj_type}_{wd}_km_surv_curves.svg"))

def plot_CYT_km_curves(CYT,true_S, cohort, outdir):
    # plot CYT only baseline
    plt.figure(figsize = (6,5))
    kmf = KaplanMeierFitter()
    adv = CYT["adverse cytogenetics"] == 1
    fav = CYT["favorable cytogenetics"] == 1
    int = CYT["intermediate cytogenetics"] == 1
    kmf.fit(true_S[adv].t, true_S[adv].e, label = f"adverse ({adv.sum()})")
    a1 = kmf.plot()
    kmf.fit(true_S[int].t, true_S[int].e, label = f"intermediate ({int.sum()})") 
    kmf.plot(ax=a1)
    kmf.fit(true_S[fav].t, true_S[fav].e, label = f"favorable ({fav.sum()})") 
    kmf.plot(ax=a1)
    a1.set_title(f"Predicted KM survival curves on cytogenetics \n {cohort}")
    a1.set_ylim([0,1])
    plt.savefig(os.path.join(outdir, f"{cohort}_CYT_km_surv_curves.svg"))
