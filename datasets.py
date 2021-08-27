import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb
import os 

class Leucegene_Public_Dataset():
    def __init__(self, CF_file, protein_coding = True) -> None:
        print("Loading ClinF file ...")
        self.CDS = protein_coding
        self.CF = pd.read_csv(CF_file, sep="\t")  # load in and preprocess Clinical Features file
        self.NS = self.CF.shape[0]
        print("Loading Gene Expression file ...")
        self.GE_TPM =  self.load_ge_tpm(cohort = "public", protein_coding = protein_coding) # load in and preprocess Gene Expression file    
        self.GE_TPM_LOG = np.log(self.GE_TPM + 1)
        
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
    
    def load_ge_tpm(self, cohort = "public", protein_coding = True):
        outfile = f"lgn_{cohort}_GE_{['TRSC','CDS'][int(protein_coding)]}_TPM.csv"
        if outfile in os.listdir("Data") :
            return pd.read_csv(f"Data/{outfile}", index_col = 0)
        else:
            print(f"TPM normalized Gene Expression (CDS only) file not found in Data/{outfile}\nNow performing CDS gene selection (if needed) + tpm norm ...")
            GE_raw_T = self.load_ge_raw().T 
            GE_raw_T["featureID_x"] = GE_raw_T.index
            GE_raw_T["featureID_y"] = GE_raw_T["featureID_x"].str.split(".", expand = True)[0].values
            # select cds
            print("Loading and assembling Gene Repertoire...")
            gene_info = self.process_gene_repertoire_data() 
            print("Processing TPM computation...")
            # merge with GE data  
            GE = GE_raw_T.merge(gene_info, on = "featureID_y")
            # filtering if needed
            if protein_coding: GE = GE[GE["gene_biotype_y"] == "protein_coding"] 
            # tpm norm
            GE_RPK = GE.iloc[:,:self.NS].astype(float) / np.array(GE["gene.length_x"]).reshape(GE.shape[0],1) / 1000
            per_million = GE_RPK.sum(0) / 1e6
            GE_TPM = GE_RPK / per_million 
            # clean up 
            # write to file 
            print(f"Writing to Data/{outfile}...")
            GE_TPM.to_csv(f"Data/{outfile}")
            return GE_TPM
    
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