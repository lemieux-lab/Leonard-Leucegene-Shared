# custom
from re import I
from engines.datasets.base_datasets import Leucegene_Dataset
import engines.datasets.FE_datasets as FE_Datasets
import engines.models.functions as functions 
import engines.models.dimredox_models as models
from engines.optimisers.base_optimisers import HPOptimiser
# base
from torch.autograd import Variable
from datetime import datetime
import pandas as pd
import pdb 
import numpy as np
from tqdm import tqdm 
import os 
import torch 
import monitoring 
import time 

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
            ds.append([cohort, Leucegene_Dataset(cohort = cohort, embedding_file = self.EMB_FILE)])
        self.datasets = dict(ds)
    
    def run_benchmarks(self):
        # fix cohort! for survival analysis
        cohort = "pronostic"
        # init results
        tst_res = []
        tr_res = [] 
        agg_c_index = []
        cohort_data = self.datasets[cohort]
        idx = np.arange(cohort_data.data["CDS"].x.shape[0])
        np.random.shuffle(idx) # shuffle dataset! 
        repn_hash = hash(sum(idx[:10] / idx[-10:]))

        # set input 
        for input in self.BENCHMARKS:
            # set data
            if input.split("-")[1] == "PCA":
                data = cohort_data.data["CDS"]
                data.generate_pca()
            else:
                data = cohort_data.data[input.split("-")[1]]
            data.reindex(idx)
            data.split_train_test(self.NFOLDS)
            # width    
            scores = [] # store risk prediction scores for agg_c_index calc
            
            # a data frame containing training optimzation results
            for foldn in tqdm(range(self.NFOLDS), desc = f"{input}"):
                print(f"foldN: {foldn + 1}")
                #pdb.set_trace()
                #data.to_DF() # reset data to cpu
                test_data = data.folds[foldn].test
                train_data = data.folds[foldn].train
                # choose model type and launch HP optim
                hpopt = HPOptimiser(train_data, 
                    model_type = input.split("-")[0], 
                    n = self.NREP_OPTIM, 
                    nfolds = self.INT_NFOLDS, 
                    nepochs = self.NEPOCHS,
                    pca_input_size_range = self.N_PCs)
                
                int_cv, opt_model = hpopt.run()
                # test
                out, l, c = opt_model._test(test_data)
                scores.append(out)
                int_cv["foldn"] = foldn + 1 
                int_cv["model_type"] = input
                int_cv["repn_hash"] = repn_hash
                tr_res.append(int_cv)
                
                #tr_res_df.to_csv(f"{self.OUTPATHS['RES']}/{input}_intcv_benchmark.csv")
                
                tst_res.append([input, foldn + 1, opt_model.params["wd"], opt_model.params["input_size"],c])
            # store training res
            tst_agg_c = functions.compute_c_index(data.y["t"], data.y["e"], np.concatenate(scores))
            agg_c_index.append(np.ones(5) * tst_agg_c)
            # compute agg_c_index
        tr_res_df = pd.concat(tr_res)
        tst_res_df = pd.DataFrame(tst_res, columns = ["model_type","foldn", "wd", "nIN_features", "fold_c_ind"])
        tst_res_df["tst_agg_c_ind"] = np.concatenate(agg_c_index)
        tst_res_df["repn_hash"] = repn_hash
        return tst_res_df, tr_res_df

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

class FE_Engine:
    def __init__(self, params) -> None:
        self.params = params

    def run_fact_emb(self):
        cohort = "pronostic"
        opt = self.params
        seed = opt.seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.manual_seed(seed)

        exp_dir = opt.load_folder
        if exp_dir is None: # we create a new folder if we don't load.
            exp_dir = monitoring.create_experiment_folder(opt)

        # creating the dataset
        print ("Getting the dataset...")
        dataset = FE_Datasets.get_dataset(opt,exp_dir)
        ds = Leucegene_Dataset(cohort).data["CDS"]

        # Creating a model
        print ("Getting the model...")

        my_model, optimizer, epoch, opt = monitoring.load_checkpoint(exp_dir, opt, dataset.dataset.input_size(), dataset.dataset.additional_info())

        # Training optimizer and stuff
        criterion = torch.nn.MSELoss()

        if not opt.cpu:
            print ("Putting the model on gpu...")
            my_model.cuda(opt.gpu_selection)

        # The training.
        print ("Start training.")
        #monitoring and predictions
        predictions =np.zeros((dataset.dataset.nb_patient,dataset.dataset.nb_gene))
        indices_patients = np.arange(dataset.dataset.nb_patient)
        indices_genes = np.arange(dataset.dataset.nb_gene)
        xdata = np.transpose([np.tile(indices_genes, len(indices_patients)),
                            np.repeat(indices_patients, len(indices_genes))])
        progress_bar_modulo = len(dataset)/100




        monitoring_dic = {}
        monitoring_dic['train_loss'] = []

        for t in range(epoch, opt.epoch):

            start_timer = time.time()

            thisepoch_trainloss = []

            with tqdm(dataset, unit="batch") as tepoch:
                for mini in tepoch:
                    tepoch.set_description(f"Epoch {t}")


                    inputs, targets = mini[0], mini[1]

                    inputs = Variable(inputs, requires_grad=False).float()
                    targets = Variable(targets, requires_grad=False).float()

                    if not opt.cpu:
                        inputs = inputs.cuda(opt.gpu_selection)
                        targets = targets.cuda(opt.gpu_selection)

                    # Forward pass: Compute predicted y by passing x to the model
                    y_pred = my_model(inputs).float()
                    y_pred = y_pred.squeeze()

                    targets = torch.reshape(targets,(targets.shape[0],))
                    # Compute and print loss

                    loss = criterion(y_pred, targets)
                    to_list = loss.cpu().data.numpy().reshape((1, ))[0]
                    thisepoch_trainloss.append(to_list)
                    tepoch.set_postfix(loss=loss.item())

                    np.save(os.path.join(exp_dir, 'pixel_epoch_{}'.format(t)),my_model.emb_1.weight.cpu().data.numpy() )
                    np.save(os.path.join(exp_dir,'digit_epoch_{}'.format(t)),my_model.emb_2.weight.cpu().data.numpy())

                    # Zero gradients, perform a backward pass, and update the weights.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            monitoring.save_checkpoint(my_model, optimizer, t, opt, exp_dir)
            monitoring_dic['train_loss'].append(np.mean(thisepoch_trainloss))
            np.save(f'{exp_dir}/train_loss.npy',monitoring_dic['train_loss'])
            functions.plot_factorized_embedding(ds, my_model.emb_2.weight.cpu().data.numpy(), 
                loss.data.cpu().numpy().reshape(1,)[0], 
                self.params.emb_size, 
                t,
                method = "TSNE",
                cohort = "pronostic"
            )