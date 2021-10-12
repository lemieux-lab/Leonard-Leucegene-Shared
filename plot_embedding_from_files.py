# other
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
import lifelines
import numpy as np
import pandas as pd
from tqdm import tqdm 
import umap
# bass
import os
import pdb
import itertools as it
import argparse
# custom 
import engines.utils as utils 
from engines.models.obo.read import read_obo 
from engines.datasets.base_datasets import Leucegene_Dataset

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-I", dest = "INPUT_FOLDER", type = str, help = "name of file to plot")
    parser.add_argument("-E", dest = "EPOCH", type = int, help = "epoch number of file to plot")
    parser.add_argument("-M", dest = "METHOD", type = str,  help = "method used TSNE/UMAP")
    parser.add_argument("-F" , dest = "FEATURE", type = str, help = "name of feature to plot")
    parser.add_argument("-O", dest = "OUTPATH", type = str, default = "PLOTS", help = "name of output path for outfile")
    args = parser.parse_args()
    return args

proj_picker = {"TSNE": TSNE, "UMAP":umap.UMAP}
def plot_factorized_embedding(ds, embedding,  e = 0, cohort = "public", method = "UMAP", feature = "Cytogenetic group", outpath="PLOT"):
    emb_size = embedding.shape[1]
    # manage colors 
    #colors = ["b", "g", "c", "y", "k","lightgreen", "darkgrey", "darkviolet"]
    r = lambda: np.random.randint(0,255)
    flatui = []
    for i in range(53):
        flatui.append('#%02X%02X%02X' % (r(),r(),r()))

    # plot cyto group
    markers = ['<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']*5
    #np.random.shuffle(markers)
    # fix label
    
    #perplexity = 30
    #data_folder = "/u/sauves/FactorizedEmbedding/run_LS_emb17/50934a761eefc71f19f324b1953fc056/"
    print(f"Epoch {e} - Plotting with {method} ...")
    
    #tsne = TSNE(n_components = 2, perplexity= perplexity, verbose =1, init = "pca")
    #proj_x = tsne.fit_transform(data)
    reducer = proj_picker[method]()
    proj_x = reducer.fit_transform(embedding)
    fig = plt.figure(figsize = (20,10))
    ds.y.columns = [c.replace(" ","_").lower() for c in  ds.y.columns]
    plt.grid(color = 'grey', linestyle = '--', linewidth = 0.5) 

    for i, cyto_group in enumerate(np.unique(ds.y[feature])):
            
        X = proj_x[ds.y[feature] == cyto_group][:,0]
        Y = proj_x[ds.y[feature] == cyto_group][:,1]
        c = flatui[i%len(flatui)]
        m = markers[i%len(markers)]
            
        plt.scatter(X, Y, label = cyto_group[:50] + "...", color = c , marker = m)
    caption = f"Leucegene {cohort} - {emb_size}-D Embedding - Epoch {e} - by {feature} - Factorized Embedding {method} -\n From CDS - With {ds.x.shape[0]} Samples and {ds.x.shape[1]} features"
    plt.title(caption)
    plt.xlabel(f"{method}- EMBEDDING")
    plt.ylabel(f"{method} - EMBEDDING")
    MIN = np.array(proj_x).flatten().min() - 1
    MAX = np.array(proj_x).flatten().max() + 1
    plt.xlim([MIN,MAX])
    plt.ylim([MIN,MAX])
    plt.box(False)
    plt.legend(bbox_to_anchor=(1, 1.0), loc='upper left')
    plt.gca().set_aspect('equal')
    o = utils.assert_mkdir(os.path.join(outpath, feature))
    fname = os.path.join(o, f"embedding{emb_size}_{method}_epoch_{e}")
    plt.savefig(f"{fname}.svg")
    plt.savefig(f"{fname}.png")
    plt.savefig(f"{fname}.pdf")

def main():
    args = parse_arguments()
    f = os.path.join(args.INPUT_FOLDER, f"digit_epoch_{args.EPOCH}.npy")
    embedding = np.load(f)
    ds = Leucegene_Dataset("pronostic", learning = False).data["CDS"]
    plot_factorized_embedding(ds, embedding, e = args.EPOCH, method = args.METHOD, outpath= args.OUTPATH, feature =args.FEATURE)
if __name__ == "__main__":
    main()