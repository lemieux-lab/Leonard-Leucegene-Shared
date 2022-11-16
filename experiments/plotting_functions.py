import matplotlib.pyplot as plt
plt.rcParams["svg.fonttype"] = "none" # text rendering in figures output 
import pdb 
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import confusion_matrix
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
    hi_risk = pred_data["pred_risk"] >= median_score
    lo_risk = pred_data["pred_risk"] < median_score
    nb_hi = int(float(nsamples / 2))
    nc_hi = (pred_data["e"][hi_risk] == 0).sum()
    nc_lo = (pred_data["e"][lo_risk] == 0).sum()
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
    
    # hi_risk = pred_data.loc[hi_risk,:][["t","e"]]
    # lo_risk = pred_data.loc[lo_risk,:][["t","e"]]
    results=logrank_test(pred_data[hi_risk].t,pred_data[lo_risk].t,event_observed_A= pred_data[hi_risk].e, event_observed_B=pred_data[lo_risk].e)
    
    model_type = HyperParams["modeltype"]
    cohort = HyperParams["cohort"]
    model_id = HyperParams["model_id"]
    input_size = HyperParams["input_size"]
    c_ind = HyperParams["c_index_metrics"]
    input_type = HyperParams["input_type"]
    surv_outpath = os.path.join(surv_curves_outdir, f"c_surv_{model_type}_{cohort}_{input_size}")
    plt.title(f"Survival curves - model_type: {model_type}")
    ax.set_xlabel(f'''timeline 
    dataset: {cohort}, Input type: {input_type}
    input_dim: {input_size} c_index: {np.round(c_ind,3)}, pvalue: {np.round(results._p_value[0], 5)}''')
    ax.grid(visible = True, alpha = 0.5, linestyle = "--")
    plt.tight_layout()
    plt.savefig(f"{surv_outpath}.svg")

def plot_correlations(corr_tbl,columns, proj_type, fig_path, figsize):
    #fig_outfile =os.path.join(fig_path, f"corr_cf_{proj_type}_{cohort}_heatmap.svg" )
    #mini_fig_outfile = os.path.join(fig_path, f"corr_cf_{proj_type}_{cohort}_heatmap_mini.svg" )
        
    # filter columns
    dat = corr_tbl[columns].iloc[::-1]
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
    plt.savefig(os.path.join(fig_path, f"6_corr_{proj_type}_CF.svg"))
    dat = dat.iloc[::-1,:]
    fig, ax = plt.subplots(figsize = figsize)
    im = ax.imshow(abs(dat), vmin = 0, vmax = 1, cmap=plt.cm.Blues)
    ax.set_yticklabels(dat.index)
    ax.set_xticks(np.arange(dat.shape[1]), minor=False)
    ax.set_yticks(np.arange(dat.shape[0]), minor=False)
    ax.xaxis.tick_top()
    ax.set_xticklabels(dat.columns, rotation = 45)
    fig.colorbar(im)
    #plt.savefig(mini_fig_outfile)
    return 

def plot_variance(pca_data, basepath):
    fig, ax = plt.subplots()
    return ax, basepath

def plot_c_surv_3_groups(cox_output, surv_curves_outdir, group_weights = [0.5, 0.5] ):
    c_scores = cox_output[1]
    pred_data = cox_output[2]
    HyperParams = cox_output[3]
    plt.figure()
    kmf = KaplanMeierFitter()
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
    input_type = HyperParams["input_type"]
    surv_outpath = os.path.join(surv_curves_outdir, f"c_surv_{cohort.replace(' ','').replace('.', '')}_{model_type}_{input_size}")
    plt.title(f"Survival curves - model_type: {model_type}")
    ax.set_xlabel(f'''timeline 
    dataset: {cohort}, input_dim: {input_size} 
    c_index: {np.round(c_ind,3)}''')
    ax.grid(visible = True, alpha = 0.5, linestyle = "--")
    plt.tight_layout()
    plt.savefig(f"{surv_outpath}.svg")

def plot_cm_any(scores1, scores2, params1, params2, outpath, group_weights):
    def bin_by_weights(scores):
        sorted_scores = np.sort(scores["pred_risk"])
        nb_fav = group_weights[0] 
        nb_int = group_weights[1]
        int_sep = sorted_scores[nb_fav]
        adv_sep = sorted_scores[nb_fav + nb_int] 
        def classify(x):
            if x < int_sep:
                return "favorable cytogenetics"
            elif x < adv_sep:
                return "intermediate cytogenetics"
            else : return "adverse cytogenetics"
        return [classify(x) for x in scores["pred_risk"]]
    scores1["group"] = bin_by_weights(scores1)
    scores2["group"] = bin_by_weights(scores2) 
    # merge two files based on index
    scores_merged = scores1.merge(scores2, left_index = True, right_index = True)
    # get confusion matrix
    scores1_cyt = scores_merged["group_x"]
    scores2_cyt = scores_merged["group_y"]

    labels = ["adverse cytogenetics", "intermediate cytogenetics", "favorable cytogenetics"]
    CM = confusion_matrix(scores1_cyt, scores2_cyt, labels = labels)
    fig, ax = plt.subplots(figsize = (12,10))
    im = ax.pcolor(CM, vmin = 0, vmax = 177, cmap=plt.cm.Blues)
    for (i, j), z in np.ndenumerate(CM):
        ax.text(j + 0.5, i + 0.5, '{}'.format(int(z)), ha='center', va='center')
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(CM.shape[1]) + 0.5, minor=False)
    ax.set_xlabel(params1["input_type"])
    ax.set_ylabel(params2["input_type"])
    ax.set_yticks(np.arange(CM.shape[0]) + 0.5, minor=False)
    
    ax.set_xticklabels(labels)
    plt.title(f"Confusion matrix: predicted: {params1['input_type']} true: {params2['input_type']}_{params1['cohort']}")
    plt.tight_layout()
    # dump
    plt.savefig(os.path.join(outpath, f"7_CM_{params1['input_type']}_{params2['input_type']}_{params1['cohort']}.svg"))
    plt.figure()
    plt.title("Predicted Risk vs predicted risk between two methods")
    plt.scatter(scores1["pred_risk"], scores2["pred_risk"])
    plt.xlabel(f'{params1["input_type"]} predicted risk')
    plt.ylabel(f'{params2["input_type"]} predicted risk')
    plt.plot([0,1], [0,1])
    plt.savefig(os.path.join(outpath, f"8_scatter_{params1['input_type']}_{params2['input_type']}_{params1['cohort']}.svg"))
    return ax

def plot_cm(pred_scores, cyt, params, outdir):
    # rename groups 
    pred_scores["Cytogenetic risk"] = [{"int":"intermediate cytogenetics", "fav":"favorable cytogenetics", "adv": "adverse cytogenetics"}[g.split(".")[0]] for g in  pred_scores["group"]]
    # merge two files based on index
    scores_merged = cyt.merge(pred_scores, left_index = True, right_index = True)
    # get confusion matrix
    true_cyt = scores_merged["Cytogenetic risk_x"]
    pred_cyt = scores_merged["Cytogenetic risk_y"]
    labels = ["adverse cytogenetics", "intermediate cytogenetics", "favorable cytogenetics"]
    CM = confusion_matrix(true_cyt, pred_cyt, labels = labels)

    # plot
    pdb.set_trace()
    fig, ax = plt.subplots(figsize = (12,10))
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
    plt.savefig(os.path.join(outdir, f"7_CM_{params['input_type']}_{params['input_size']}_{params['modeltype']}_{params['cohort']}.svg"))
    return ax

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
