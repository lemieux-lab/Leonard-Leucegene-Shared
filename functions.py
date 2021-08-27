from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
import numpy as np
import pdb
import matplotlib.pyplot as plt 
import utils
from tqdm import tqdm 
import os
def run_pca(dataset, outpath):
    print("Running PCA...")
    # init object
    pca = PCA()
    # fit to data
    pca.fit(dataset.GE_TPM_LOG)
    # get loadings
    # 
    # transform in self.NS dimensions
    # 
    # Writes to file 
    # 
    return pca  

def run_tsne(dataset):
    print("Running TSNE...")
    tsne = TSNE(perplexity=np.random.randint(15,30), init = "pca", verbose = 1) # random perplexity
    proj_data = tsne.fit_transform(dataset.GE_TPM_LOG.T)
    return proj_data, tsne

def run_tsne_plotting(dataset, proj_data, tsne, outpath, cohort, protein_coding, mode):
    outpath = utils.assert_mkdir(os.path.join(outpath, "TSNE", cohort,['TRSC','CDS'][int(protein_coding)] ))
    markers = [".", "v",">", "<","^","p", "P"]
    colors = ["b", "g", "c", "y", "k","m", "darkgrey", "darkviolet"]
    for feature in tqdm(['Project_ID_NGS', 'FAB classification', 'Cytogenetic group', 'Cytogenetic risk', 'Status at sampling', 'Age at sampling', 'Sex', 'WHO classification', 'Tissue']):
        fig = plt.figure(figsize = (20,10))
        plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5)     
        np.random.shuffle(markers)
        np.random.shuffle(colors)    
        for i, level in enumerate(np.unique(dataset.CF[feature])):
            c = colors[i%len(colors)]
            m = markers[i%len(markers)]
            X = proj_data[:,0][dataset.CF[feature] == level]
            Y = proj_data[:,1][dataset.CF[feature] == level]
            
            plt.scatter(X, Y, color = c, marker=m, label = str(level).replace(" or ", "/\n"))
        # Some aesthetics
        plt.xlabel("TSNE-1")
        plt.ylabel("TSNE-2")
        caption = f"Leucegene - {cohort} - by {feature} - With Gene Expression t-SNE -\n From {['Whole Transcriptome','Protein Coding'][int(protein_coding)]} - With {dataset.NS} Samples and {dataset.GE_TPM_LOG.shape[0]} features"
        # center text
        plt.title(caption)
        plt.box(False)
        plt.legend(bbox_to_anchor=(.9, 1.0), loc='upper left')
        MIN = np.array(proj_data).flatten().min() - 1
        MAX = np.array(proj_data).flatten().max() + 1
        plt.xlim([MIN,MAX])
        plt.ylim([MIN,MAX])
        plt.gca().set_aspect('equal')
        fname = f"{outpath}/lgn_{cohort}_GE_TSNE_{['TRSC','CDS'][int(protein_coding)]}_TPM_{feature}"
        plt.savefig(f"{fname}.svg")
        plt.savefig(f"{fname}.png")
        plt.savefig(f"{fname}.pdf")