# imports
import numpy as np
import pdb
from sklearn.decomposition import PCA
import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
from engines.utils import assert_mkdir
import os
def remove_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
        
def experiment_1():
# FOND OUT THERE WERE DUPLICATE SYMBOLS !!! CHANGE TO ENSMBL ID ASAP 
    from engines.datasets.base_datasets import Leucegene_Dataset
    data = Leucegene_Dataset("pronostic").data["CDS"]
    X = data.x
    cols = X.columns 
    N = 10
    f, axes = plt.subplots(figsize=(13, 6.5 * N), ncols = 2, nrows = N)
    for i in range(N):
        indices = np.arange(len(cols))
        np.random.shuffle(indices)
        M = len(cols)
        k = int(M/2)
        c1 = X.iloc[:,indices[k:]] # split dataset 
        c2 = X.iloc[:,indices[:k]]
        # PCA
        pca1 = PCA()
        proj_x1 = pca1.fit_transform(c1)
        pca2 = PCA()
        proj_x2 = pca2.fit_transform(c2)

        RMSE_1_2 = np.sqrt(((proj_x1[:,:2]  - proj_x2[:,:2]) ** 2).sum())
        print(RMSE_1_2)
        
        D1 = pd.DataFrame(proj_x1[:,:2], columns = ["PC1", "PC2"])
        D2 = pd.DataFrame(proj_x2[:,:2], columns = ["PC1", "PC2"])
        #axes[i, 0].scatter(D1["PC1"], D1["PC2"])
        #remove_frame(axes[i,0])
        #axes[i,0].set_title(f"D1 - RMSE(D2) = {round(RMSE,3)}")
        #axes[i, 1].scatter(D2["PC1"], D2["PC2"])
        #remove_frame(axes[i,1])
        #axes[i,1].set_title(f"D2 - RMSE(D1) = {round(RMSE,3)}")
        outdir = assert_mkdir("RES/FIGS/PCA")
        
        axes[i, 0].set_title(f"D1, D2 RMSE(D1[PC1],D2[PC2]) = {round(RMSE_1_2,3)}") 
        axes[i, 0].set_xlabel("PC1")
        axes[i, 0].set_ylabel("PC2")
        axes[i, 0].grid(color = 'grey', linestyle = '--', linewidth = 0.5) 
        axes[i, 0].scatter(D1["PC1"], D1["PC2"], alpha = 0.5, edgecolors = "k", label = "D1")
        axes[i, 0].scatter(D2["PC1"], D2["PC2"], alpha = 0.5, edgecolors = "k", label = "D2")
        
        RMSE_2_3 = np.sqrt(((proj_x1[:,1:3]  - proj_x2[:,1:3]) ** 2).sum())
        print(RMSE_2_3)
        
        D1 = pd.DataFrame(proj_x1[:,1:3], columns = ["PC2", "PC3"])
        D2 = pd.DataFrame(proj_x2[:,1:3], columns = ["PC2", "PC3"])
        #axes[i, 0].scatter(D1["PC1"], D1["PC2"])
        #remove_frame(axes[i,0])
        #axes[i,0].set_title(f"D1 - RMSE(D2) = {round(RMSE,3)}")
        #axes[i, 1].scatter(D2["PC1"], D2["PC2"])
        #remove_frame(axes[i,1])
        #axes[i,1].set_title(f"D2 - RMSE(D1) = {round(RMSE,3)}")
        outdir = assert_mkdir("RES/FIGS/PCA")
        
        axes[i, 1].set_title(f"D1, D2 RMSE(D1[PC1],D2[PC2]) = {round(RMSE_2_3,3)}") 
        axes[i, 1].set_xlabel("PC2")
        axes[i, 1].set_ylabel("PC3")
        axes[i, 1].grid(color = 'grey', linestyle = '--', linewidth = 0.5) 
        axes[i, 1].scatter(D1["PC2"], D1["PC3"], alpha = 0.5, edgecolors = "k", label = "D1")
        axes[i, 1].scatter(D2["PC2"], D2["PC3"], alpha = 0.5, edgecolors = "k", label = "D2")
    
    
    plt.savefig(os.path.join(outdir, f"PCA_c1_c2_N={N}.png"))




    pdb.set_trace()
    # do splitting
    # do pca 
    # plot data 
    # evaluate difference 

def main():
    experiment_1()

if __name__=="__main__":
    main()