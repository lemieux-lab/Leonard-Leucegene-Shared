import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb
import os 

class Data:
    def __init__(self,x, y ) -> None:
        self.x = x
        self.y = y

class Leucegene_Dataset():
    def __init__(self, CF_file) -> None:
        print("Loading ClinF file ...")
        self.w = ["CDS", "TRSC"]
        self.C = cohort
        self._CDS = w == "CDS" 
        self.CF = pd.read_csv(CF_file, sep="\t")  # load in and preprocess Clinical Features file
        self.NS = self.CF.shape[0]
        print("Loading Gene Expression file ...")
        self._load_ge_tpm() # load in and preprocess Gene Expression file    
        self.set_data()
    
    def set_data(self):

        if self.w == "CDS":
            
            # select cds
            print("Loading and assembling Gene Repertoire...")
            self.gene_info = self.process_gene_repertoire_data() 
            ### select based on repertoire
            
            # filtering if needed
            # merge with GE data  
            self._GE_CDS_TPM = self._GE_raw_T.merge(self.gene_info[self.gene_info["gene_biotype_y"] == "protein_coding"], on = "featureID_y")
            
            self.data = Data(np.log(self._GE_CDS_TPM + 1), self.CF)
            
        if self.w == "TRSC":
            self.data = Data(np.log(self._GE_TPM + 1), self.CF)
        
        self.NF = self.data.x.shape[0]
    
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
        outfile = f"lgn_{self.C}_GE_{self.w}_TPM.csv"
        if outfile in os.listdir("Data") :
            self._GE_TPM = pd.read_csv(f"Data/{outfile}", index_col = 0)
        else:
            print(f"TPM normalized Gene Expression (CDS only) file not found in Data/{outfile}\nNow performing CDS gene selection (if needed) + tpm norm ...")
            self._GE_raw_T = self.load_ge_raw().T 
            self._GE_raw_T["featureID_x"] = self._GE_raw_T.index
            self._GE_raw_T["featureID_y"] = self._GE_raw_T["featureID_x"].str.split(".", expand = True)[0].values
            
            print("Processing TPM computation...")
            # tpm norm
            GE_RPK = self._GE_raw_T.iloc[:,:self.NS].astype(float) / np.array(self._GE_raw_T["gene.length_x"]).reshape(GE.shape[0],1) / 1000
            per_million = GE_RPK.sum(0) / 1e6
            self._GE_TPM =  GE_RPK / per_million 
            # clean up 
            # write to file 
            print(f"Writing to Data/{outfile}...")
            self._GE_TPM.to_csv(f"Data/{outfile}")
    
    def load_ge_raw(self, cohort = "public"):
        outfile = f"lgn_{cohort}_GE.assembled.csv"
        if outfile in os.listdir("Data") :
            print(f"Loading Raw Gene Expression file from {outfile}...")
            return pd.read_csv(f"Data/{outfile}", index_col = 0)
        else : 
            
            print(f"Gene Expression file not found... in Data/{outfile} \nLoading {self.NS} samples GE readcounts from files ...")
            samples = []
            for i, sample in tqdm(enumerate(self.CF.sample_id)): 
                samples.append( pd.read_csv(f"/u/leucegene/data/sample/{sample}/transcriptome/readcount/star_GRCh38/star_genes_readcount.unstranded.xls", sep = "\t", index_col=0).T) 
            print("Concatenating ...")
            df = pd.concat(samples)
            print(f"writing to Data/{outfile} ...")
            df.to_csv(f"Data/{outfile}")
            return df 