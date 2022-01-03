import matplotlib.pyplot as plt
import pdb 
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import os

def plot_hi_risk_lo_risk(risk_scores, data, fig_outdir, proj_type, cohort, perfo_list):
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
    plt.savefig(os.path.join(fig_outdir, f"{cohort}_{proj_type}_km_surv_curves.svg"))

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
