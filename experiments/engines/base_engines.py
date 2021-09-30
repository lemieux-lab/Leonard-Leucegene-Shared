# custom
from re import I
from experiments.engines.datasets.datasets import Leucegene_Dataset
import models
# base
from datetime import datetime
import utils 
import functions
import pandas as pd
import pdb 
import numpy as np
from tqdm import tqdm 
import os 

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
        self.N_PCs = params.N_PCs
        # HARDCODE
        self.NFOLDS = 5
        self.INT_NFOLDS = 5
        self._init_CF_files()
        self._load_datasets()
        # HARDCODE
        self.OUTPATHS = {   # dict
            "RES": os.path.join("RES") #utils.assert_mkdir(f"RES{datetime.now()}"))
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
        # init results
        tst_res = []
        # set input 
        for input in self.BENCHMARKS:
            # set data
            # width
            width = "TRSC" if input.split("-")[1] == "LSC17" else "CDS"
            data = self.datasets[cohort].data[width]
            data.set_input_targets(input, embfile = self.EMB_FILE)
            data.shuffle() # IMPORTANT
            data.split_train_test(nfolds = self.NFOLDS) # 5 fold cross-val
            scores = [] # store risk prediction scores for agg_c_index calc
            tr_res = [] # a data frame containing training optimzation results
            for foldn in tqdm(range(self.NFOLDS)):
                print(f"foldN: {foldn + 1}")
                #pdb.set_trace()
                #data.to_DF() # reset data to cpu
                test_data = data.folds[0].test
                train_data = data.folds[0].train
                
                # choose model type and launch HP optim
                int_cv, opt_model = models.hpoptim(train_data, 
                    model_type = input.split("-")[0], 
                    n = self.NREP_OPTIM, 
                    nfolds = self.INT_NFOLDS, 
                    nepochs = self.NEPOCHS,
                    input_size_range = self.N_PCs)
                
                # test
                out, l, c = opt_model._test(test_data)
                scores.append(out)
                if "CPHDNN" in input: epochs = opt_model.params["epochs"]
                else: epochs = -999
                int_cv["foldn"] = foldn + 1 
                tr_res.append(int_cv)
                tr_res_df = pd.concat(tr_res)
                tr_res_df.to_csv(f"{self.OUTPATHS['RES']}/{input}_intcv_benchmark.csv")
                
                tst_res.append([input, foldn + 1, opt_model.params["wd"], opt_model.params["input_size"],c])
                print(tst_res[-1])
            # compute agg_c_index
            tst_agg_c = functions.compute_aggregated_c_index(scores, data)
            tst_res = pd.DataFrame(tst_res, columns = ["model_type","foldn", "wd", "nIN_PCs", "fold_c_ind"])
            tst_res["tst_agg_c_ind"] = tst_agg_c
            tst_res.to_csv(f"{self.OUTPATHS['RES']}/{input}_test_results.csv")
            pdb.set_trace()

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

    def run_fact_emb(self):
        # load Gene Expression data
        # run Fact_emb
        # save embedding 
        # plot embedding
        pass