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

def generate_tsne(datasets, cohort, width, outpath):
    print("Running TSNE...")
    tsne = TSNE(perplexity=np.random.randint(15,30), init = "pca", verbose = 1) # random perplexity
    proj_data = tsne.fit_transform(datasets[cohort].data[width].x)
    return proj_data, tsne

def tsne_plotting(datasets, proj_data, tsne_data, cohort, width, outpath):
    outpath = utils.assert_mkdir(os.path.join(outpath, cohort, width ))
    markers = [".", "v",">", "<","^","p", "P"]
    colors = ["b", "g", "c", "y", "k","m", "darkgrey", "darkviolet"]
    features = ["WHO classification","Sex", "Induction_Type",'HSCT_Status_Type' ,'Cytogenetic risk', 'FAB classification','Tissue', 'RNASEQ_protocol',"IDH1-R132 mutation" ,'FLT3-ITD mutation',  "NPM1 mutation"]
    for feature in tqdm(features):
        fig = plt.figure(figsize = (20,10))
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)     
        np.random.shuffle(markers)
        np.random.shuffle(colors) 
        for i, level in enumerate(np.unique(datasets[cohort].data[width].y[feature])):
            c = colors[i%len(colors)]
            m = markers[i%len(markers)]
            X = proj_data[:,0][datasets[cohort].data[width].y[feature] == level]
            Y = proj_data[:,1][datasets[cohort].data[width].y[feature] == level]
            
            plt.scatter(X, Y, color = c, marker=m, label = str(level).replace(" or ", "/\n"))
        # Some aesthetics
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        caption = f"Leucegene - {cohort} - by {feature} - With Gene Expression t-SNE p={tsne_data.perplexity}-\n From {width} - With {datasets[cohort].NS} Samples and {datasets[cohort].data[width].x.shape[1]} features"
        # center text
        plt.title(caption)
        plt.box(False)
        plt.legend(bbox_to_anchor=(.9, 1.0), loc='upper left')
        MIN = np.array(proj_data).flatten().min() - 1
        MAX = np.array(proj_data).flatten().max() + 1
        plt.xlim([MIN,MAX])
        plt.ylim([MIN,MAX])
        plt.gca().set_aspect('equal')
        fname = f"{outpath}/lgn_{cohort}_GE_TSNE_{width}_TPM_{feature}"
        plt.savefig(f"{fname}.svg")
        plt.savefig(f"{fname}.png")
        plt.savefig(f"{fname}.pdf")