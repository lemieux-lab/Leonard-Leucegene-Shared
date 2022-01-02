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

def _merge_files(input_fpath, dataset):
    frames = []
    for file in os.listdir(os.path.join(input_fpath,dataset)):
        fname = os.path.join(input_fpath,dataset, file)
        frames.append(pd.read_csv(fname))
    return pd.concat(frames)

def hist_by_input_d(data, dataset, input_dims, proj_types, by_input_dims_output_path):
    binwidth = 0.002
    for in_d in tqdm(input_dims):
            in_d_data = data[data["input_dims"] == in_d]    
            plt.figure()
            descr_dict = {
                "RPgauss_var": "Random projection (gaussian)",
                "RSelect": "Random selection",
                "PCA": "Principal Component Analysis",
                "CF-PCA": f"Clinical features (8) + PCA ({in_d})"
            }
            frames = []
            for proj_type in np.unique(in_d_data.proj_type):
                frames.append(in_d_data.c_ind_tst[(in_d_data["proj_type"] == proj_type)][:1000])
            hist_data = pd.DataFrame(np.array(frames).T, columns= np.unique(in_d_data.proj_type))
            plt.hist(hist_data, bins = np.arange(min(data.c_ind_tst), max(data.c_ind_tst) + binwidth, binwidth), density=1, histtype='bar', stacked=True, label = proj_types)
            plt.legend()
            plt.title(f"Performance with input dim = {in_d} with {dataset}")
            plt.ylabel("Count / density")
            plt.xlim([0.5, max(data.c_ind_tst)])
            plt.xlabel("Performance (concordance index)")
            #axes[0].set_xticks(np.arange(0,100, 5))
            #axes[0].set_xlim([0,method_data["input_d"].max() + 3])
            #axes[0].set_ylim([0.5, 0.75])
            plt.savefig(os.path.join(by_input_dims_output_path, f"hist_{in_d}.svg"))

def plot_method_vs_method_by_input_pval(data, input_dims, methods, by_methods_output_path): 
    fig, axes = plt.subplots(figsize = (16, 8), nrows = 4, ncols = 4)
    for i, method_i in enumerate(methods):
        for j, method_j in enumerate(methods):
            if method_i is not method_j:
                I = data[data["proj_type"] == method_i]
                J = data[data["proj_type"] == method_j]
                c_mean_I = I.groupby("input_d").mean("c_ind_tst").reset_index()
                c_mean_J = J.groupby("input_d").mean("c_ind_tst").reset_index()
                
                
                axes[i,j].set_title(method_j)
                axes[i,j].set_xlabel(" input dim" )
                axes[i,j].set_ylabel(method_i + " c index diff" )
                axes[i,j].set_ylim([-0.2,0.2])
                axes[i,j].hlines(y = 0, xmin =0, xmax=max(input_dims), color = "k", linestyle = "--")
                for input_d in input_dims:
                    c_I = I[I["input_d"] == input_d]
                    c_J = J[J["input_d"] == input_d]
                    t_stat, p = ttest_ind( c_J.c_ind_tst, c_I.c_ind_tst)
                    pval_color = "k"
                    if p < 0.001:
                        if t_stat / abs(t_stat) == -1: pval_color = "r"
                        else : pval_color = "b"
                    axes[i,j].scatter([input_d], [c_J.c_ind_tst.mean() - c_I.c_ind_tst.mean()], c = pval_color)
        plt.tight_layout()
        plt.savefig(os.path.join(by_methods_output_path, "p_val_by_input_d_methods.svg"))

def plot_LSC17_vs_method_by_input_pval(data, input_dims, methods, by_methods_output_path): 
    fig, axes = plt.subplots(figsize = (16, 2), nrows = 1, ncols = 4)
    I = data[data["proj_type"] == "LSC17"]
    c_mean_I = I.groupby("input_d").mean("c_ind_tst").reset_index()
    c_I = I
    for j, method in enumerate(methods):
        J = data[data["proj_type"] == method]
        c_mean_J = J.groupby("input_d").mean("c_ind_tst").reset_index()
        axes[j].set_title(method)
        axes[j].set_xlabel("input dim" )
        axes[j].set_ylabel("vs LSC17 c index diff" )
        axes[j].set_ylim([-0.2,0.2])
        axes[j].hlines(y = 0, xmin =0, xmax=max(input_dims), color = "k", linestyle = "--")
                
        for input_d in input_dims: 
            c_J = J[J["input_d"] == input_d]
            t_stat, p = ttest_ind(c_I.c_ind_tst,  c_J.c_ind_tst)
            pval_color = "k"
            if p < 0.001:
                if t_stat / abs(t_stat) == -1: pval_color = "r"
                else : pval_color = "b"
            axes[j].scatter([input_d], [ c_I.c_ind_tst.mean() - c_J.c_ind_tst.mean()], c = pval_color)
    plt.tight_layout()
    plt.savefig(os.path.join(by_methods_output_path, "p_val_by_input_d_lsc17.svg"))

def run(args):
    pass 

def outadated(args):
    plt.rcParams['svg.fonttype'] = 'none'
    # list outputfiles / directories
    base_fig_path = os.path.join("RES", "FIG", f"EXP_{args.EXP}")
    datasets = ["lgn_pronostic"]
    proj_types = ["CF-PCA", "PCA","RSelect", "RPgauss_var"]
    input_dims = np.arange(1,50)
    for dataset in datasets:
        by_methods_output_path = utils.assert_mkdir(os.path.join(base_fig_path, dataset,"by_methods"))
        by_input_dims_output_path = utils.assert_mkdir(os.path.join(base_fig_path, dataset, "by_input_dim"))
        # do two separate analysis for different datasets
        # merge files 
        input_fpath = os.path.join("RES", "EXP_1")
        data = _merge_files(input_fpath, dataset)
        data = data[data.proj_type.isin(proj_types)]
        real_input_dims = np.array(data["input_d"])
        real_input_dims[data.proj_type == "CF-PCA"] = real_input_dims[data.proj_type == "CF-PCA"] + 8
        data["input_dims"] = real_input_dims 
        print(f"Plotting c_index vs input_dims / method ... --> {by_methods_output_path}")
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
        print("Plotting median curves ...")
        input_dim_median_data = data.groupby(["proj_type","input_dims"]).median().reset_index()
        plt.figure()
        for proj_type in proj_types:
            d = input_dim_median_data[ input_dim_median_data["proj_type"] == proj_type]
            plt.plot(d["input_dims"].astype(int),d["c_ind_tst"], label = proj_type)
        plt.scatter([17] , [data[data["proj_type"] == "LSC17"].c_ind_tst.median()], marker="x", s = 15, color= 'k', label = "LSC17 median")
        plt.ylim([0.5, 0.75])
        plt.legend()
        plt.savefig(os.path.join(base_fig_path, dataset, "median_scores_by_inp_dim.svg"))

        # line plots overfitting (c ind test and train)
        print("Plotting overfitting curves ...")
        for proj_type in proj_types:
            plt.figure()
            d = input_dim_median_data[ input_dim_median_data["proj_type"] == proj_type]
            plt.plot(d["input_dims"],d["c_ind_tr"], label = "training")
            plt.plot(d["input_dims"],d["c_ind_tst"], label = "test")
            plt.ylim([0.5, 1])
            plt.legend()
            plt.savefig(os.path.join(by_methods_output_path, f"overfitting_med_scores_{proj_type}.svg"))

        # make 1 plot c_index by mythod / input_dims
        
        print("Plotting histograms by input_d")
        hist_by_input_d(data, dataset, input_dims, proj_types, by_input_dims_output_path)
        
        print("Plotting LSC17 histogram")
        plt.figure()
        hist_data = data[data["proj_type"] == "LSC17"].c_ind_tst
        label = f"LSC17, n= {hist_data.shape[0]}"
        plt.hist(hist_data, alpha = 0.5, label = label, bins = 30)
        plt.vlines(x = np.mean(hist_data),ymin = 0 , ymax = 5, linestyle = "--")
        plt.text(np.mean(hist_data), 5, '{:0.2f}'.format(np.mean(hist_data)))
        plt.legend()
        plt.title(f"LSC17 Performance with {dataset}")
        plt.ylabel("Count / density")
        plt.xlim([0.5, max(data.c_ind_tst)])
        plt.xlabel("Performance (concordance index)")
        plt.savefig(os.path.join(by_input_dims_output_path, "LSC17_hist.svg"))

        
        #print("Plotting method vs method by input_d p_val difference")
        #plot_method_vs_method_by_input_pval(data, input_dims, proj_types, by_methods_output_path)
        
        print("Plotting LSC17 vs method by input_d p_val difference")
        plot_LSC17_vs_method_by_input_pval(data, input_dims,  proj_types, by_methods_output_path)