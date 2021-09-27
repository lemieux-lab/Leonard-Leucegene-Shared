# custom
from re import I
from datasets import Leucegene_Dataset
# base
from datetime import datetime
import utils 
import functions
import pandas as pd
import pdb 
import numpy as np
import models
from tqdm import tqdm 

class Engine:
    def __init__(self, params):
        self.OUTFILES = params.OUTFILES # bool
        self.RUN_PCA = params.PCA # bool
        self.RUN_TSNE = params.TSNE # bool
        self.N_TSNE = params.N_TSNE # int
        self.MAX_PC = params.MAX_PC # int
        self.RUN_PLOTTING = params.PLOT # bool
        self.COHORTS = params.COHORTS # str list
        self.WIDTHS = params.WIDTHS # str list
        self.RUN_GO = params.GO # bool
        self.GO_TOP_N = params.GO_TOP_N # int
        self.RUN_CPH = params.CPH
        self.RUN_CPHDNN = params.CPHDNN
        self.BENCHMARKS = params.BENCHMARKS
        self.EMB_FILE = params.EMB_FILE
        self.NREP_OPTIM = params.NREP_OPTIM
        self.NEPOCHS = params.NEPOCHS
        # HARDCODE
        self.NFOLDS = 5
        self.INT_NFOLDS = 5
        self._init_CF_files()
        self._load_datasets()
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

    def _load_datasets(self):
        ds = []
        for cohort in self.COHORTS:
            ds.append([cohort, Leucegene_Dataset(cohort = cohort,embedding_file =self.EMB_FILE)])
        self.datasets = dict(ds)
    
    def run_benchmarks(self):
        # fix cohort and width
        cohort = "pronostic"
        # width
        width = "CDS"
        # init results
        res = []
        # set input 
        for input in self.BENCHMARKS:
            # set data
            data = self.datasets[cohort].data[width]
            data.set_input_targets(input = input.split("-")[1], filter = input.split("-")[0] == "CPH")
            data.shuffle()
            data.split_train_test(nfolds = self.NFOLDS) # 5 fold cross-val
            for foldn in tqdm(range(self.NFOLDS)):
                print(f"foldN: {foldn + 1}")
                #pdb.set_trace()
                #data.to_DF() # reset data to cpu
                test_data = data.folds[0].test
                train_data = data.folds[0].train
                
                # choose model type and launch HP optim
                int_cv, opt_model = models.hpoptim(train_data, model_type = input.split("-")[0], n = self.NREP_OPTIM, nfolds = self.INT_NFOLDS, nepochs = self.NEPOCHS)
                # test
                out, l, c = opt_model._test(test_data)
                if "CPHDNN" in input: epochs = opt_model.params["epochs"]
                else: epochs = -999
                res.append([foldn, input, self.NREP_OPTIM, self.NFOLDS, self.INT_NFOLDS, epochs,c])
            int_cv.to_csv(f"{self.OUTPATHS['RES']}/{input}_intcv_benchmark.csv")
            # inference
            # out = model.forward(test_data.x)
            # c_index = utils.concordance_index(out, test_data.y["t"], test_data.y["e"])
            # res.append([input, 1, 5, 10, 500, c_index])

        res = pd.DataFrame(res, columns = ["foldn", "model_type","nrep_optim",  "external_cv","internal_cv", "n_epochs", "c_index"])
        res.to_csv(f"{self.OUTPATHS['RES']}/run_benchmarks.csv")
        print(res) 
    def run_visualisations(self):
        for cohort in self.COHORTS:
            for width in self.WIDTHS: 
                # generate_pca 
                if self.RUN_PCA:
                    self.PCA = self.datasets[cohort].data[width].generate_pca()

                    if self.RUN_GO:
                        self.GO = functions.get_ontologies(self.datasets, 
                        self.PCA,
                        cohort = cohort,
                        width = width,
                        outpath = self.OUTPATHS["RES"],
                        max_pc = self.MAX_PC,
                        top_n = self.GO_TOP_N)
                    # plot pca
                    if self.RUN_PLOTTING:
                        functions.pca_plotting(self.datasets,
                        self.PCA,
                        cohort = cohort,
                        width = width,
                        base_outpath = self.OUTPATHS["RES"]
                        )
                # generate ontologies pca
                
                # run CPH_DNN
                if self.RUN_CPH:
                    # run PCA + CPH
                    pca_cph = models.train_test(self.datasets[cohort].data[width], "CPH", input= "pca")
                    # run CLINF + CPH
                    clinf_cph = models.train_test(self.datasets[cohort].data[width],"CPH", input = "clinf")
                    # run PCA + CLINF + CPH 
                    pca_clinf_cph = models.train_test(self.datasets[cohort].data[width], "CPH", input = "clinf+pca")
                 
                if self.RUN_CPHDNN:
                    # run PCA + CPHDNN
                    pca_cphdnn = models.train_test(self.datasets[cohort].data[width], "CPHDNN", input  = "pca")
                    # run CLINF + CPHDNN
                    pca_clinf_cphdnn = models.train_test(self.datasets[cohort].data[width], "CPHDNN", input = "clinf+pca")
                    if self.FIXED_EMB:
                        # run FACT_EMB + CPHDNN
                        print("Not yet implemented ...")
                                                         
                # generate TSNE
                if self.RUN_TSNE: 
                    self.tsne_data = functions.generate_tsne(self.datasets, 
                    cohort = cohort, 
                    width = width, 
                    outpath = self.OUTPATHS["RES"],
                    N = self.N_TSNE)
                    # plot TSNE
                    if self.RUN_PLOTTING: 
                        functions.tsne_plotting(self.datasets,
                    self.tsne_data, 
                    cohort, 
                    width, 
                    self.OUTPATHS["RES"])

