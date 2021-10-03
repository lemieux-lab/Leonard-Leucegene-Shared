# imports
import numpy as np
import pdb
from sklearn.decomposition import PCA
import seaborn as sns 
import matplotlib.pyplot as plt

def experiment_1():
# FOND OUT THERE WERE DUPLICATE SYMBOLS !!! CHANGE TO ENSMBL ID ASAP 
    from engines.datasets.base_datasets import Leucegene_Dataset
    X = Leucegene_Dataset("pronostic").data["CDS"].x
    cols = X.columns 
    for i in range(10):
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

        RMSE = np.sqrt(((proj_x1  - proj_x2) ** 2).sum())
        print(RMSE)
        
        pdb.set_trace()




    pdb.set_trace()
    # do splitting
    # do pca 
    # plot data 
    # evaluate difference 

def main():
    experiment_1()

if __name__=="__main__":
    main()