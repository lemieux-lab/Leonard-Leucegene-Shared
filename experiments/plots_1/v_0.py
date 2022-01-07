from tqdm import tqdm 
from scipy.stats import ttest_ind
from scipy.stats import t
import os 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
from engines import utils 
import pdb

def _merge_files(input_fpath, dataset):
    frames = []
    for file in os.listdir(os.path.join(input_fpath,dataset)):
        fname = os.path.join(input_fpath,dataset, file)
        frames.append(pd.read_csv(fname))
    return pd.concat(frames)

def hist_by_input_d(data, dataset, input_dims, by_input_dims_output_path):
    print("Plotting histograms by input_d")
    binwidth = 0.002
    for in_d in tqdm(input_dims):
        in_d_data = data[data["input_dims"] == in_d]    
        plt.figure()
        labels = [
        "Clinical features + PCA",
        "Principal Component Analysis",
        "Random projection (gaussian)",
        "Random selection"
        ]

        ## filter by proj_type
        CF_PCA = in_d_data.c_ind_tst[in_d_data["proj_type"] == "CF-PCA"]
        PCA = in_d_data.c_ind_tst[in_d_data["proj_type"] == "PCA"]
        RP = in_d_data.c_ind_tst[in_d_data["proj_type"] == "RPgauss_var"]
        RS = in_d_data.c_ind_tst[in_d_data["proj_type"] == "RSelect"]
        def w (l):
            return [1 / l.shape[0]] * l.shape[0] if l.shape[0] else [0] * l.shape[0]
        plt.hist((CF_PCA, PCA, RP, RS), 
        weights = (w(CF_PCA), w(PCA), w(RP), w(RS)), 
        stacked =True, bins = 30, density = True, label = labels)
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
    print("Plotting method vs method by input_d p_val difference")
    fig, axes = plt.subplots(figsize = (16, 8), nrows = 4, ncols = 4)
    for i, method_i in enumerate(methods):
        for j, method_j in enumerate(methods):
            if method_i is not method_j:
                I = data[data["proj_type"] == method_i]
                J = data[data["proj_type"] == method_j]
                c_mean_I = I.groupby("input_dims").mean("c_ind_tst").reset_index()
                c_mean_J = J.groupby("input_dims").mean("c_ind_tst").reset_index()


                axes[i,j].set_title(method_j)
                axes[i,j].set_xlabel(" input dim" )
                axes[i,j].set_ylabel(method_i + " c index diff" )
                axes[i,j].set_ylim([-0.2,0.2])
                axes[i,j].hlines(y = 0, xmin =0, xmax=max(input_dims), color = "k", linestyle = "--")
                for input_d in input_dims:
                    c_I = I[I["input_dims"] == input_d]
                    c_J = J[J["input_dims"] == input_d]
                    t_stat, p = ttest_ind( c_J.c_ind_tst, c_I.c_ind_tst)
                    pval_color = "k"
                    if p < 0.001:
                        if t_stat / abs(t_stat) == -1: pval_color = "r"
                        else : pval_color = "b"
                    axes[i,j].scatter([input_d], [c_J.c_ind_tst.mean() - c_I.c_ind_tst.mean()], c = pval_color)
        plt.tight_layout()
        plt.savefig(os.path.join(by_methods_output_path, "p_val_by_input_d_methods.svg"))

def plot_LSC17_vs_method_by_input_pval(data, input_dims, methods, by_methods_output_path): 
    print("Plotting LSC17 vs method by input_d p_val difference")
    fig, axes = plt.subplots(figsize = (16, 2), nrows = 1, ncols = 4)
    I = data[data["proj_type"] == "LSC17"]
    c_mean_I = I.groupby("input_dims").mean("c_ind_tst").reset_index()
    c_I = I
    for j, method in enumerate(methods):
        J = data[data["proj_type"] == method]
        c_mean_J = J.groupby("input_dims").mean("c_ind_tst").reset_index()
        axes[j].set_title(method)
        axes[j].set_xlabel("input dim" )
        axes[j].set_ylabel("vs LSC17 c index diff" )
        axes[j].set_ylim([-0.2,0.2])
        axes[j].hlines(y = 0, xmin =0, xmax=max(input_dims), color = "k", linestyle = "--")

        for input_d in input_dims: 
            c_J = J[J["input_dims"] == input_d]
            t_stat, p = ttest_ind(c_I.c_ind_tst,  c_J.c_ind_tst)
            pval_color = "k"
            if p < 0.001:
                if t_stat / abs(t_stat) == -1: pval_color = "r"
                else : pval_color = "b"
            axes[j].scatter([input_d], [ c_I.c_ind_tst.mean() - c_J.c_ind_tst.mean()], c = pval_color)
    plt.tight_layout()
    plt.savefig(os.path.join(by_methods_output_path, "p_val_by_input_d_lsc17.svg"))

def boxplots(data, proj_types, dataset, by_methods_output_path):
    print(f"Plotting c_index vs input_dims / method ... --> {by_methods_output_path}")
    ## make 1 plot c_index vs input_dims / method
    for proj_type in proj_types:
        method_data = data[data.proj_type == proj_type]
        plt.figure()
        axes = method_data[["c_ind_tst", "input_d"]].boxplot(by="input_d", grid = False, showfliers = False, return_type = 'axes')
        axes[0].set_title(f"Performance of {proj_type} with {dataset}")
        axes[0].set_xlabel("Number of input features")
        axes[0].set_ylabel("Performance (concordance index)")
        axes[0].set_xticks(np.arange(0,100, 5))
        axes[0].set_xlim([0,100])
        axes[0].set_ylim([0.5, 0.75])
        plt.savefig(os.path.join(by_methods_output_path, f"boxplots_{proj_type}.svg"))

def line_plots_median(data, input_dim_median_data, proj_types, dataset, base_fig_path):
    print("Plotting median curves ...")
    
    plt.figure()
    for proj_type in proj_types:
        d = input_dim_median_data[ input_dim_median_data["proj_type"] == proj_type]
        plt.plot(d["input_dims"].astype(int),d["c_ind_tst"], label = proj_type)
    plt.scatter([17] , [data[data["proj_type"] == "LSC17"].c_ind_tst.median()], marker="x", s= 15, color= 'k', label = "LSC17 median")
    plt.ylim([0.5, 0.75])
    plt.legend()
    plt.savefig(os.path.join(base_fig_path, dataset, "median_scores_by_inp_dim.svg"))

def line_plots_overfitting(input_dim_median_data, proj_types, by_methods_output_path):
    print("Plotting overfitting curves ...")
    for proj_type in proj_types:
        plt.figure()
        d = input_dim_median_data[ input_dim_median_data["proj_type"] == proj_type]
        plt.plot(d["input_dims"],d["c_ind_tr"], label = "training")
        plt.plot(d["input_dims"],d["c_ind_tst"], label = "test")
        plt.ylim([0.5, 1])
        plt.legend()
        plt.savefig(os.path.join(by_methods_output_path, f"overfitting_med_scores_{proj_type}.svg"))

def lsc17_hist(data, dataset, by_input_dims_output_path):
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

def run(args):
    plt.rcParams['svg.fonttype'] = 'none'
    # list outputfiles / directories
    base_fig_path = args.OUTPATH
    dataset = args.COHORT
    proj_types = ["CF-PCA", "PCA","RSelect", "RPgauss_var"]
    input_dims = np.arange(args.INPUT_DIMS[0], args.INPUT_DIMS[1])
    ## outpaths
    by_methods_output_path = utils.assert_mkdir(os.path.join(base_fig_path, dataset,"by_methods"))
    by_input_dims_output_path = utils.assert_mkdir(os.path.join(base_fig_path, dataset, "by_input_dim"))
    ## merge files and parse  
    input_fpath = os.path.join("RES", "EXP_1")
    data = _merge_files(input_fpath, dataset)
    data = data[data.proj_type.isin(np.concatenate([proj_types, ["LSC17"]]))]
    real_input_dims = np.array(data["input_d"])
    real_input_dims[data.proj_type == "CF-PCA"] = real_input_dims[data.proj_type == "CF-PCA"] + 8
    data["input_dims"] = real_input_dims 
    ## box plots
    # boxplots(data, proj_types, dataset, by_methods_output_path)
    ## prepare grouped dataset
    input_dim_median_data = data.groupby(["proj_type","input_dims"]).median().reset_index()
    ## line plots c_ind_tst (median)
    line_plots_median(data, input_dim_median_data, proj_types, dataset, base_fig_path)
    ## line plots overfitting (c ind test and train)
    # line_plots_overfitting(input_dim_median_data, proj_types, by_methods_output_path)
    ## make 1 plot c_index by mythod / input_dims
    # hist_by_input_d(data, dataset, input_dims, by_input_dims_output_path)
    ## histogram of lsc17 perfo alone
    # lsc17_hist(data, dataset, by_input_dims_output_path)
    ## plot methods vs methods diff and pval
    # plot_method_vs_method_by_input_pval(data, input_dims, proj_types, by_methods_output_path)
    ## plot LSC17 vs other methods diff and pval
    # plot_LSC17_vs_method_by_input_pval(data, input_dims,  proj_types, by_methods_output_path)