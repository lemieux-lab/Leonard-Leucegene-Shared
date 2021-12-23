#imports 
from numpy.lib.arraysetops import intersect1d
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import engines.utils as utils
import os
import pdb 

def _merge_files(input_fpath, dataset):
    frames = []
    for file in os.listdir(os.path.join(input_fpath,dataset)):
        fname = os.path.join(input_fpath,dataset, file)
        frames.append(pd.read_csv(fname))
    return pd.concat(frames)

def run(args):
    plt.rcParams['svg.fonttype'] = 'none'
    # list outputfiles / directories
    base_fig_path = os.path.join("RES", "FIG", f"EXP_{args.EXP}")
    datasets = ["lgn_pronostic", "tcga_target_aml"]
    proj_types = ["PCA","RSelect", "RPgauss"]
    input_dims = np.arange(1,80)
    for dataset in datasets:
        by_methods_output_path = utils.assert_mkdir(os.path.join(base_fig_path, dataset,"by_methods"))
        by_input_dims_output_path = utils.assert_mkdir(os.path.join(base_fig_path, dataset, "by_input_dim"))
        # do two separate analysis for different datasets
        # merge files 
        input_fpath = os.path.join("RES", "EXP_1")
        data = _merge_files(input_fpath, dataset)
        # make 1 plot c_index vs input_dims / method
        for proj_type in proj_types:
            method_data = data[data.proj_type == proj_type]
            # box plots
            plt.figure()
            axes = method_data[["c_ind_tst", "input_d"]].boxplot(by="input_d", grid = False, showfliers = False, return_type = 'axes')
            axes[0].set_title(f"Performance of {proj_type} with {dataset}")
            axes[0].set_xlabel("Number of input features")
            axes[0].set_ylabel("Performance (concordance index)")
            axes[0].set_xticks(np.arange(0,100, 5))
            axes[0].set_xlim([0,100])
            axes[0].set_ylim([0.5, 0.75])
            plt.savefig(os.path.join(by_methods_output_path, f"boxplots_{proj_type}.svg"))
        
        # line plots c_ind_tst (median)

        input_dim_median_data = data.groupby(["proj_type","input_d"]).median().reset_index()
        plt.figure()
        for proj_type in proj_types:
            d = input_dim_median_data[ input_dim_median_data["proj_type"] == proj_type]
            plt.plot(d["input_d"],d["c_ind_tst"], label = proj_type)
        plt.ylim([0.5, 0.75])
        plt.legend()
        plt.savefig(os.path.join(base_fig_path, dataset, "median_scores_by_inp_dim.svg"))

        # line plots overfitting (c ind test and train)
        for proj_type in proj_types:
            plt.figure()
            d = input_dim_median_data[ input_dim_median_data["proj_type"] == proj_type]
            plt.plot(d["input_d"],d["c_ind_tr"], label = "training")
            plt.plot(d["input_d"],d["c_ind_tst"], label = "test")
            plt.ylim([0.5, 1])
            plt.legend()
            plt.savefig(os.path.join(by_methods_output_path, f"overfitting_med_scores_{proj_type}.svg"))

        # make 1 plot c_index by mythod / input_dims
        bw = 0.02
        for in_d in input_dims:
            in_d_data = data[data.input_d == in_d]    
            plt.figure()
            descr_dict = {
                "RPgauss": "Random projection (gaussian)",
                "RSelect": "Random selection",
                "PCA": "Principal Component Analysis"
            }
            for proj_type in ["RPgauss", "RSelect", "PCA"]:
                hist_data = in_d_data[in_d_data["proj_type"] == proj_type].c_ind_tst
                label = f"{descr_dict[proj_type]}, n= {hist_data.shape[0]}"
                plt.hist(hist_data, alpha = 0.5, label = label, bins = 30)
            plt.legend()
            plt.title(f"Performance with input dim = {in_d} with {dataset}")
            plt.ylabel("Count / density")
            plt.xlim([0.5, max(data.c_ind_tst)])
            plt.xlabel("Performance (concordance index)")
            #axes[0].set_xticks(np.arange(0,100, 5))
            #axes[0].set_xlim([0,method_data["input_d"].max() + 3])
            #axes[0].set_ylim([0.5, 0.75])
            plt.savefig(os.path.join(by_input_dims_output_path, f"hist_{in_d}.svg"))
        # outputs to files
    
    fig_outfile =  os.path.join(basepath, f"summary_table_cf_{cohort}.svg")
    print(f"Action 2: Printing summary data table of clinical infos. --> {outfile, fig_outfile}")