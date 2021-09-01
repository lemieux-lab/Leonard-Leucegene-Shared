# custom
from datasets import Leucegene_Dataset
# base
from datetime import datetime
import utils 
import functions
import pandas as pd
import pdb 
import numpy as np

class Engine:
    def __init__(self, params):
        self.OUTFILES = params.OUTFILES # bool
        self.RUN_PCA = params.PCA # bool
        self.RUN_TSNE = params.TSNE # bool
        self.RUN_PLOTTING = params.PLOT # bool
        self.COHORTS = params.COHORTS
        self.WIDTHS = params.WIDTHS
        self._init_CF_files()
        self.load_datasets()
        # HARDCODE
        self.OUTPATHS = {   # dict
            "RES": utils.assert_mkdir(f"RES{datetime.now()}"),
        }
    def _init_CF_files(self):
        infos = pd.read_csv("Data/lgn_ALL_CF", sep = "\t").T
        infos.columns = infos.iloc[0,:] # rename cols
        infos = infos.iloc[1:,:] # remove 1st row
        features = ["Prognostic subset", "Age_at_diagnosis", 
        "Sex", "Induction_Type",
        'HSCT_Status_Type' ,'Cytogenetic risk', 
        'FAB classification','Tissue', 
        'RNASEQ_protocol',"IDH1-R132 mutation" ,
        "Relapse",'FLT3-ITD mutation', 
        "WHO classification", "NPM1 mutation", 
        "Overall_Survival_Time_days", "Overall_Survival_Status",
        "Cytogenetic group"] # select features
        infos = infos[features]
        for cohort in ["lgn_public", "lgn_pronostic"]:
            samples = pd.read_csv(f"Data/{cohort}_samples", index_col = 0)
            CF_file = infos.merge(samples, left_index = True, right_on = "sampleID")
            CF_file.index = CF_file.sampleID
            CF_file = CF_file[np.setdiff1d(CF_file.columns, ["sampleID"])]
            CF_file.to_csv(f"Data/{cohort}_CF")

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

