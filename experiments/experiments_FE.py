import pandas as pd
import numpy as np
import pdb
import datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os 
import umap
## load in data
ds = datasets.Leucegene_Dataset("pronostic").data["CDS"]
ds_numpy = ds.x.to_numpy()
#with open("/u/sauves/FactorizedEmbedding/data/lgn_pronostic_GE_CDS_TPM_LOG.npy", 'wb') as o:
#    np.save(o, ds_numpy)

r = lambda: np.random.randint(0,255)

flatui = []
for i in range(53):
    flatui.append('#%02X%02X%02X' % (r(),r(),r()))

# plot cyto group
markers = ['<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']*5

#colors = ["b", "g", "c", "y", "k","lightgreen", "darkgrey", "darkviolet"]
feature = "Cytogenetic group"
#perplexity = 30
data_folder = "/u/sauves/FactorizedEmbedding/run_LS_emb17/50934a761eefc71f19f324b1953fc056/"
files = [f for f in os.listdir(data_folder) if "digit" in f]
np.random.shuffle(markers)
method = "UMAP"
#np.random.shuffle(colors) 
for file in files:
    e = int(file.split(".")[0].split("_")[2])
    print(f"Epoch {e} - file: {file}")
    data = np.load(os.path.join(data_folder, file))
    #tsne = TSNE(n_components = 2, perplexity= perplexity, verbose =1, init = "pca")
    #proj_x = tsne.fit_transform(data)
    reducer = umap.UMAP()
    proj_x = reducer.fit_transform(data)
    fig = plt.figure(figsize = (20,10))
    for i, cyto_group in enumerate(np.unique(ds.y[feature])):
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)     
        X = proj_x[ds.y[feature] == cyto_group][:,0]
        Y = proj_x[ds.y[feature] == cyto_group][:,1]
        c = flatui[i%len(flatui)]
        m = markers[i%len(markers)]
        
        plt.scatter(X, Y, label = cyto_group[:50] + "...", color = c , marker = m)
    caption = f"Leucegene - Pronostic - {data.shape[1]}D Embedding - Epoch {e} - by {feature} - Factorized Embedding {method} -\n From CDS - With {ds.x.shape[0]} Samples and {ds.x.shape[1]} features"
    plt.title(caption)
    plt.xlabel(f"{method}- EMBEDDING")
    plt.ylabel(f"{method} - EMBEDDING")
    MIN = np.array(proj_x).flatten().min() - 1
    MAX = np.array(proj_x).flatten().max() + 1
    plt.xlim([MIN,MAX])
    plt.ylim([MIN,MAX])
    plt.box(False)
    plt.legend(bbox_to_anchor=(-0.5, 1.0), loc='upper left')
    plt.gca().set_aspect('equal')
    plt.savefig(f"FIGS/proj_by_epoch_FE/embedding17_{method}_epoch_{e}.svg")
    plt.savefig(f"FIGS/proj_by_epoch_FE/embedding17_{method}_epoch_{e}.png")
    plt.savefig(f"FIGS/proj_by_epoch_FE/embedding17_{method}_epoch_{e}.pdf")
pdb.set_trace() 
