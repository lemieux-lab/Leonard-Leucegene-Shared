from operator import xor
import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb
import os 
from sklearn.decomposition import PCA 
import torch

class Data:
    def __init__(self,x, y ,gene_info, name = "data", reindex = True, device = "cpu") -> None:
        self.name = name
        self.x = x
        self.y = y
        self.gene_info = gene_info
        self.device = device
        if reindex: self._reindex_targets()
    
    def folds_to_cuda_tensors(self, device = "cuda:0"):
        if "cuda" in self.device : return 
        for i in range(len(self.folds)):
            train_x = torch.Tensor(self.folds[i].train.x.values).to(device)
            train_y = torch.Tensor(self.folds[i].train.y.values).to(device)
            test_x = torch.Tensor(self.folds[i].test.x.values).to(device)
            test_y = torch.Tensor(self.folds[i].test.y.values).to(device)
            train = Data(x = train_x, y = train_y, gene_info = None, reindex = False)
            test = Data(x = test_x, y = test_y, gene_info = None, reindex =False)
            self.folds[i].train = train
            self.folds[i].test = test
    
    def to(self, device):    
        if "cuda" in self.device: return 
        self.x = torch.Tensor(self.x.values).to(device)
        self.y = torch.Tensor(self.y.values).to(device)
        self.device = device

    def to_DF(self):
        if "cuda" in self.device :
            pdb.set_trace()
            self.x = pd.DataFrame(self.x.detach().cpu().numpy())
            self.y = pd.DataFrame(self.y.detach().cpu().numpy(), columns = ["t", "e"])
    
    def generate_pca(self):
        print("Running PCA...")
        
        # init object
        self._pca = PCA()
        # fit to data
        self._xpca = pd.DataFrame(self._pca.fit_transform(self.x), index = self.x.index)
        # get loadings
        # 
        # transform in self.NS dimensions
        # 
        # Writes to file 
        # 
        return {"proj_x":self._xpca, "pca":self._pca }
    def LSC17(self):
        
        lsc17 = pd.read_csv("Data/SIGNATURES/LSC17_expressions.csv", index_col = 0)
        #LSC17_expressions = self.x[self.x.columns[self.x.columns.isin(lsc17.merge(self.gene_info, left_on = "ensmbl_id_version", right_on = "featureID_x").featureID_y)]]
        return lsc17

    def set_input_targets(self, input, embfile = None):
        self.model_type = input.split("-")[0]
        self.data_type = input.split("-")[1]
        # input
        if self.data_type == "LSC17":
            # do something
            self.x = self.LSC17()
        
        if self.data_type == "PCA":
            try: 
                self.x = self._xpca
            except Exception:
                self.generate_pca()
                self.x = self._xpca

        elif self.data_type == "FE":
            print("Fetching embedding file...")
            if embfile.split(".")[-1] == "npy":
                emb_x = np.load(embfile)
                self._xemb = emb_x
            elif embfile.split(".")[-1] == "csv":
                emb_x = pd.read_csv(embfile, index_col=0)
                self._xemb = emb_x[emb_x.index.isin(self.y.index)]
            self.x = pd.DataFrame(self._xemb, index = self.x.index)
            # evaluate variance in cols, drop low variance ones
            #if self.model_type == "CPH":
            #    nrem  = (self.x.var(0) < 0.001).sum()
            #    self.x = self.x.T[(self.x.var(0) > 0.001)].T
            #    print(f"Removed {nrem} columns with low variance")
            
        elif self.data_type == "clinf":
            self.x = self.y[np.setdiff1d(self.y.columns, ["Overall_Survival_Time_days", "Overall_Survival_Status"])] # do smthing
        else :
            self.x = self.x 

        # targets
        if len(self.y.columns) > 2:
            self.y = self.y[["Overall_Survival_Time_days", "Overall_Survival_Status"]]
            self.y.columns = ["t", "e"]
        
    def shuffle(self):
        self.x = self.x.sample(frac = 1)
        self._reindex_targets()
        
    def split_train_test(self, nfolds):
        if "cuda" in self.device: return # do nothing if dataset is already split! 
        dummy_ds = self.x.sample(frac = 1)
        
        n = self.x.shape[0]
        fold_size = int(float(n)/nfolds)
        self.folds = []
        for i in range(nfolds):
            fold_ids = np.arange(i * fold_size, min((i + 1) * fold_size, n))
            test_x = dummy_ds.iloc[fold_ids,:]
            test_y = self.y.loc[test_x.index]
            train_x = self.x.loc[~self.x.index.isin(test_x.index)]
            train_y = self.y.loc[train_x.index]
            self.folds.append(Data(self.x, self.y,  self.gene_info))
            self.folds[i].train = Data(train_x, train_y, self.gene_info)
            self.folds[i].test = Data(test_x,test_y, self.gene_info)


    def create_shuffles(self, n):
        print(f"Creates {n} data shuffles ...")
        self.shuffles = [self.x.sample(frac =1).index for i in range(n)]
    
    def select_shuffle(self, n):
        self.x = self.x.loc[self.shuffles[n]]
        self._reindex_targets()

    def _reindex_targets(self):
        self.y = self.y.loc[self.x.index]


class Leucegene_Dataset():
    def __init__(self, cohort, embedding_file = None) -> None:
        self.COHORT = cohort
        print(f"Loading ClinF {self.COHORT} file ...")
        self.CF_file = f"Data/lgn_{self.COHORT}_CF"
        self.EMB_FILE = embedding_file
        self.CF = pd.read_csv(self.CF_file, index_col = 0)  # load in and preprocess Clinical Features file
        self.NS = self.CF.shape[0]
        print("Loading Gene Expression file ...")
        self._load_ge_tpm() # load in and preprocess Gene Expression file    
        self._set_data()
     
    
    def _set_data(self):
          
        # select cds
        print("Loading and assembling Gene Repertoire...")
        self.gene_info = self.process_gene_repertoire_data() 
        ### select based on repertoire
        # filtering if needed, merge with GE data  
        self._GE_CDS_TPM = self._GE_TPM.merge(self.gene_info[self.gene_info["gene_biotype_y"] == "protein_coding"], left_index = True, right_on = "featureID_y")
        # clean up
        self._GE_CDS_TPM.index = self._GE_CDS_TPM.SYMBOL
        self._GE_CDS_TPM = (self._GE_CDS_TPM.iloc[:,:-self.gene_info.shape[1]]).T
        self.GE_CDS_LOG = np.log(self._GE_CDS_TPM + 1)
        self.GE_TRSC_LOG = np.log(self._GE_TPM.T + 1)
        # set CDS data
        cds_data = Data(self.GE_CDS_LOG, self.CF, self.gene_info, name = f"{self.COHORT}_CDS")
        # set TRSC data
        trsc_data = Data(self.GE_TRSC_LOG, self.CF, self.gene_info, name = f"{self.COHORT}_TRSC") 
        
        self.data = {"CDS": cds_data, "TRSC": trsc_data, }


    def dump_infos(self, outpath):
        # self.GE_CDS_TPM_LOG.to_csv(f"{outpath}/GE_CDS_TPM_LOG.csv")
        pass
    def process_gene_repertoire_data(self):
        # load in Gencode 37 repertoire (NOTE: no biotype present !!) 
        Gencode37 = pd.read_csv("/u/leucegene/data/Homo_sapiens.GRCh38_H32/annotations/Homo_sapiens.GRCh38.Gencode37.genes.tsv", sep = "\t")
        # load in Esembl99 repertoire (NOTE: ensembl id incomplete (no version id))
        Ensembl99 = pd.read_csv("/u/leucegene/data/Homo_sapiens.GRCh38_H32/annotations/Homo_sapiens.GRCh38.Ensembl99.genes.tsv", sep = "\t")
        # extract gene infos and store
        gene_info = Gencode37.merge(Ensembl99, on = "SYMBOL") 
        return gene_info
    
    def _load_ge_tpm(self):
        outfile = f"lgn_{self.COHORT}_GE_TRSC_TPM.csv"
        if outfile in os.listdir("Data") :
            self._GE_TPM = pd.read_csv(f"Data/{outfile}", index_col = 0)
        else:
            print(f"TPM normalized Gene Expression (CDS only) file not found in Data/{outfile}\nNow performing tpm norm ...")
            self._GE_raw_T = self.load_ge_raw().T 
            self._GE_raw_T["featureID_x"] = self._GE_raw_T.index
            self._GE_raw_T["featureID_y"] = self._GE_raw_T["featureID_x"].str.split(".", expand = True)[0].values
            
            print("Processing TPM computation...")
            # get gene infos
            gene_info = self.process_gene_repertoire_data()
            self._GE = self._GE_raw_T.merge(gene_info, on = "featureID_y") 
            gene_lengths = np.matrix(self._GE["gene.length_x"]).T / 1000 # get gene lengths in KB
            # tpm norm
            GE_RPK = self._GE.iloc[:,:self.NS].astype(float) / gene_lengths 
            per_million = GE_RPK.sum(0) / 1e6
            self._GE_TPM =  GE_RPK / per_million 
            # clean up 
            self._GE_TPM.index = self._GE.featureID_y
            # write to file 
            print(f"Writing to Data/{outfile}...")
            self._GE_TPM.to_csv(f"Data/{outfile}")
    
    def load_ge_raw(self):
        
        outfile = f"lgn_{self.COHORT}_GE.assembled.csv"
        if outfile in os.listdir("Data") :
            print(f"Loading Raw Gene Expression file from {outfile}...")
            return pd.read_csv(f"Data/{outfile}", index_col = 0)
        else : 
            print(f"Gene Expression file not found... in Data/{outfile} \nLoading {self.NS} samples GE readcounts from files ...")
            samples = []
            for sample in tqdm(self.CF.index): 
                samples.append( pd.read_csv(f"/u/leucegene/data/sample/{sample}/transcriptome/readcount/star_GRCh38/star_genes_readcount.unstranded.xls", sep = "\t", index_col=0).T) 
            print("Concatenating ...")
            df = pd.concat(samples)
            print(f"writing to Data/{outfile} ...")
            df.to_csv(f"Data/{outfile}")
            return df 