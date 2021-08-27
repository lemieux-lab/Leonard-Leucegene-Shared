from datetime import datetime 
from functions import *

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