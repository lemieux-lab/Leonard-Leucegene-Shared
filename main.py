# custom
import utils 
from datasets import Leucegene_Public_Dataset
# base
import pdb
from sklearn.decomposition import PCA
from datetime import datetime

def run_pca(dataset, outpath):
    # init object
    pca = PCA()
    # fit to data
    pca.fit(dataset.GE_CDS_TPM_LOG)
    # get loadings
    # 
    # transform in self.NS dimensions
    # 
    # Writes to file 
    # 
    return pca  
    
class Engine:
    def __init__(self, args, dataset):
        self.OUTFILES = args.OUTFILES 
        self.PCA = args.PCA
        self.TSNE = args.TSNE
        self.PLOT = args.PLOT
        self.dataset = dataset

        self.OUTPATHS = {
            "RES": f"RES{datetime.now()}",
        }

    def run(self):
        # outputs info
        self.dataset.dump_infos(self.OUTPATHS["RES"])
        # computes pca
        self.PCA = run_pca(self.dataset, self.OUTPATHS["RES"])
            # writes table 
        # computes tsne
            # writes table
        # scatter plot 
            # by features
            # by PC 1...10 x PC 1...10

        pass

def main():
    eng = Engine(
        args = utils.parse_arguments(), 
        dataset = Leucegene_Public_Dataset(
            CF_file = "Data/lcg_public_CF"))
    # load leucegene_public dataset
    eng.run()
    pdb.set_trace()
    
if __name__ == "__main__":
    main()
