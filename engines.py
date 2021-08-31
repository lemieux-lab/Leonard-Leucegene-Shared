# custom
from datasets import Leucegene_Dataset
# base
from datetime import datetime
import utils 
import functions

class Engine:
    def __init__(self, params):
        self.OUTFILES = params.OUTFILES # bool
        self.RUN_PCA = params.PCA # bool
        self.RUN_TSNE = params.TSNE # bool
        self.RUN_PLOTTING = params.PLOT # bool
        self.COHORTS = params.COHORTS
        self.WIDTHS = params.WIDTHS
        self.load_datasets()
        # HARDCODE
        self.OUTPATHS = {   # dict
            "RES": utils.assert_mkdir(f"RES{datetime.now()}"),
        }
    def load_datasets(self):
        ds = []
        for cohort in self.COHORTS:
            ds.append([cohort, Leucegene_Dataset(cohort = cohort)])
        self.datasets = dict(ds)

    def run(self):
        for cohort in self.COHORTS:
            for width in self.WIDTHS: 
                # generate_pca 
                self.PCA = functions.generate_pca(self.datasets, 
                    cohort = cohort, 
                    width = width, 
                    outpath = self.OUTPATHS["RES"])
                # plot pca
                # generate ontologies pca

                # generate TSNE
                self.proj_data, self.tsne_data = functions.generate_tsne(self.datasets, 
                    cohort = cohort, 
                    width = width, 
                    outpath = self.OUTPATHS["RES"])
                # plot TSNE
                functions.tsne_plotting(self.datasets, 
                    self.proj_data, 
                    self.tsne_data, 
                    cohort, 
                    width, 
                    self.OUTPATHS["RES"])

