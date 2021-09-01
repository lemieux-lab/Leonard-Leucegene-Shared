import matplotlib
matplotlib.use('Agg')
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
import numpy as np
import pdb
import matplotlib.pyplot as plt 
import utils
from tqdm import tqdm 
import os
def generate_pca(datasets, cohort, width, outpath):
    print("Running PCA...")
    # init object
    pca = PCA()
    # fit to data
    proj_data = pca.fit_transform(datasets[cohort].data[width].x)
    # get loadings
    # 
    # transform in self.NS dimensions
    # 
    # Writes to file 
    # 
    return proj_data, pca  

def generate_tsne(datasets, cohort, width, outpath, N =1):
    
    data = []
    p = np.random.randint(15,30) # select perplexity
    for n in range(N): 
        # shuffle data
        datasets[cohort].data[width].shuffle()
        print(f"Running TSNE Rep {n +1} ...")
        tsne_engine = TSNE(perplexity=p, init = "pca", verbose = 1) # random perplexity
        proj_x = tsne_engine.fit_transform(datasets[cohort].data[width].x)
        data.append({"proj_x": proj_x, "tsne": tsne_engine})
    return data

def tsne_plotting(datasets, tsne_data, cohort, width, outpath):
    outpath = utils.assert_mkdir(os.path.join(outpath, cohort, width ))
    markers = [".", "v",">", "<","^","p", "P"]
    colors = ["b", "g", "c", "y", "k","lightgreen", "darkgrey", "darkviolet"]
    features = ["WHO classification","Sex", "Cytogenetic group","Induction_Type",'HSCT_Status_Type' ,'Cytogenetic risk', 'FAB classification','Tissue', 'RNASEQ_protocol',"IDH1-R132 mutation" ,'FLT3-ITD mutation',  "NPM1 mutation"]
    print("Plotting figures ...")
    for rep_n, tsne in enumerate(tsne_data):
        for feature in tqdm(features, desc =f"PLOTTING REP: [{rep_n +1}] ..."):
            fig = plt.figure(figsize = (20,10))
            plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)     
            np.random.shuffle(markers)
            np.random.shuffle(colors) 
            for i, level in enumerate(np.unique(datasets[cohort].data[width].y[feature])):
                c = colors[i%len(colors)]
                m = markers[i%len(markers)]
                X = tsne["proj_x"][:,0][datasets[cohort].data[width].y[feature] == level]
                Y = tsne["proj_x"][:,1][datasets[cohort].data[width].y[feature] == level]
                
                plt.scatter(X, Y, color = c, marker=m, label = str(level).replace(" or ", "/\n"))
            # Some aesthetics
            plt.xlabel("TSNE-1")
            plt.ylabel("TSNE-2")
            caption = f"Leucegene - {cohort} - by {feature} - With Gene Expression t-SNE p={tsne['tsne'].perplexity} rep={rep_n+1} -\n From {width} - With {datasets[cohort].NS} Samples and {datasets[cohort].data[width].x.shape[1]} features"
            # center text
            plt.title(caption)
            plt.box(False)
            plt.legend(bbox_to_anchor=(.9, 1.0), loc='upper left')
            MIN = np.array(tsne["proj_x"]).flatten().min() - 1
            MAX = np.array(tsne["proj_x"]).flatten().max() + 1
            plt.xlim([MIN,MAX])
            plt.ylim([MIN,MAX])
            plt.gca().set_aspect('equal')
            fname = f"{outpath}/lgn_{cohort}_GE_TSNE_{width}_TPM_{feature}_[{rep_n+1}]"
            plt.savefig(f"{fname}.svg")
            plt.savefig(f"{fname}.png")
            plt.savefig(f"{fname}.pdf")