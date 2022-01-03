#imports 
from numpy.lib.arraysetops import intersect1d
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import engines.utils as utils
import os
import pdb 
from tqdm import tqdm 
from scipy.stats import ttest_ind
from scipy.stats import t
from engines import utils 
def run(args):
    # plot survival curves
    from experiments import plotting_functions as plt_f
    # fix cohort name
    cohort = "lgn_pronostic"
    # set basepath
    scores_data = pd.read_csv(os.path.join("RES", "EXP_2", "action5", f"{cohort}_scores_by_method.csv"))
    perfo_data = pd.read_csv(os.path.join("RES", "EXP_2", "action5", f"{cohort}_bootstrap_10000_by_method.csv"))
    outdir = utils.assert_mkdir(os.path.join("RES", "FIGS", "FIG1"))
    # load results data 
    # plot
    plt_f.km_survival_curves(scores_data, perfo_data, outdir, cohort)
    