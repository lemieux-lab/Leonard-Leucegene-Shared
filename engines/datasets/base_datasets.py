from operator import xor
from re import I
import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb
import os 
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import random_projection
import torch
from engines import utils
import xml.etree.ElementTree as ET

class Data:
    def __init__(self,x, y ,gene_info, name = "data", reindex = True, device = "cpu", learning = True) -> None:
        self.name = name
        self.x = x
        self.y = y
        self.gene_info = gene_info
        self.device = device
        if reindex: self._reindex_targets()
        if learning and len(self.y.columns) > 2:
            self.y = self.y[["Overall_Survival_Time_days", "Overall_Survival_Status"]]
            self.y.columns = ["t", "e"]
    
    def clone(self):
        return Data(self.x, self.y, self.gene_info)
      
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

    def split_train_test(self, nfolds, device = "cpu"):
         # do nothing if dataset is already split! 
        self.x = self.x.sample(frac = 1)
        
        n = self.x.shape[0]
        fold_size = int(float(n)/nfolds)
        self.folds = []
        for i in range(nfolds):
            fold_ids = np.arange(i * fold_size, min((i + 1) * fold_size, n))
            test_x = self.x.iloc[fold_ids,:]
            test_y = self.y.loc[test_x.index]
            train_x = self.x.loc[~self.x.index.isin(test_x.index)]
            train_y = self.y.loc[train_x.index]
            self.folds.append(Data(self.x, self.y,  self.gene_info))
            self.folds[i].train = Data(train_x, train_y, self.gene_info)
            self.folds[i].train.to(device)
            self.folds[i].test = Data(test_x,test_y, self.gene_info)
            self.folds[i].test.to(device)
        # reorder original data
        self._reindex_targets()

    def to(self, device):    
        self.device = device
        if device == "cpu": return
        self.x = torch.Tensor(self.x.values).to(device)
        self.y = torch.Tensor(self.y.values).to(device)
        
    def to_DF(self):
        if "cuda" in self.device :
            pdb.set_trace()
            self.x = pd.DataFrame(self.x.detach().cpu().numpy())
            self.y = pd.DataFrame(self.y.detach().cpu().numpy(), columns = ["t", "e"])
    
    def generate_PCA(self):
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
        self.x = self._xpca
        return {"proj_x":self._xpca, "pca":self._pca }
        
    def generate_RP(self, method, n = 17):
        print("Running Random Projection...")
        if method == "gauss":
            self.transformer = random_projection.GaussianRandomProjection(n_components=n)
            self._xrp = pd.DataFrame(self.transformer.fit_transform(self.x), index = self.x.index)
        elif method == "sparse":
            self.transformer = random_projection.SparseRandomProjection(n_components=n)
            self._xrp = pd.DataFrame(self.transformer.fit_transform(self.x), index = self.x.index)
        self.x = self._xrp
    
    def generate_RS(self, n):
        print("Generating Random signature (genes with var > 0.1)...")
        high_v_cols = self.x.columns[self.x.var() > 0.1]
        col_ids = np.arange(len(high_v_cols))
        np.random.shuffle(col_ids)
        # assert enough variance
        self.x = self.x[high_v_cols[:n]]

    def generate_SVD(self, n):
        print("Running Singular Value Decomposition SVD ...")
        svd = TruncatedSVD(n_components = n)
        self.x = pd.DataFrame(svd.fit_transform(self.x), index = self.x.index)

    def shuffle(self):
        self.x = self.x.sample(frac = 1)
        self._reindex_targets()

    def remove_unexpressed_genes(self, verbose = 0):
        """ removes all genes with no expression across all samples"""
        d = self.x.shape[1]
        n_rm = self.x.sum(0) != 0
        self.x = self.x.loc[:,n_rm]
        if verbose:
            print(f"removed {d - n_rm.sum()} genes with null expression across samples ")
            print(f"Now datataset hase shape {self.x.shape}")


    def create_shuffles(self, n):
        print(f"Creates {n} data shuffles ...")
        self.shuffles = [self.x.sample(frac =1).index for i in range(n)]
    
    def select_shuffle(self, n):
        self.x = self.x.loc[self.shuffles[n]]
        self._reindex_targets()
    def reindex(self, idx):
        self.x = self.x.iloc[idx]
        self.y = self.y.loc[self.x.index]

    def _reindex_targets(self):
        self.y = self.y.loc[self.x.index]
    
    def _reindex_expressions(self):
        self.x = self.x.loc[self.y.index]

class SurvivalGEDataset():
    def __init__(self) -> None:
        self.gene_repertoire = self.process_gene_repertoire_data()
    
    def process_gene_repertoire_data(self):
        print("Loading and assembling Gene Repertoire...")
        # load in Gencode 37 repertoire (NOTE: no biotype present !!) 
        Gencode37 = pd.read_csv("/u/leucegene/data/Homo_sapiens.GRCh38_H32/annotations/Homo_sapiens.GRCh38.Gencode37.genes.tsv", sep = "\t")
        # load in Esembl99 repertoire (NOTE: ensembl id incomplete (no version id))
        Ensembl99 = pd.read_csv("/u/leucegene/data/Homo_sapiens.GRCh38_H32/annotations/Homo_sapiens.GRCh38.Ensembl99.genes.tsv", sep = "\t")
        # extract gene infos and store
        gene_info = Gencode37.merge(Ensembl99, on = "SYMBOL") 
        return gene_info
    
    def load_dataset(self,cohort):
        if cohort == "tcga_target_aml":
            return self.load_tcga_target_aml()
        elif cohort == "lgn_pronostic":
            return self.load_lgn_pronostic()    
    
    def load_tcga_target_aml(self):
        tcga = TCGA_Dataset(self.gene_repertoire)
        return tcga

    def load_lgn_pronostic(self):
        lgn = Leucegene_Dataset(self.gene_repertoire, cohort = "pronostic")
        return lgn

class TCGA_Dataset():
    def __init__(self, gene_repertoire):
        self.data_path = "Data"
        self.tcga_data_path = os.path.join(self.data_path, "TCGATARGETAML")
        self.tcga_manifests_path = os.path.join(self.tcga_data_path, "MANIFESTS")
        self.tcga_counts_path = os.path.join(self.tcga_data_path, "COUNTS")
        self.tcga_cd_path  = os.path.join(self.tcga_data_path, "CLINICAL")
        
        self.data = self.assemble_load_tcga_data()
        self.gene_repertoire = gene_repertoire

    def assemble_load_tcga_data(self):
        ## FETCH GENE EXPRESSIONS ##
        GE_manifest_file = os.path.join(self.tcga_manifests_path, 'gdc_manifest.2020-07-23_GE.txt')
        if not self.assert_load_from_manifest(GE_manifest_file, self.tcga_counts_path):
            self.load_tcga_aml(GE_manifest_file, target_dir = self.tcga_counts_path)
        else : print('OUT TCGA + TARGET -AML Gene Expression Raw data already loaded locally on disk')
        
        ## FETCH CLINICAL DATA ##
        CD_manifest_file = os.path.join(self.tcga_manifests_path, 'gdc_manifest.2020-07-23_CD.txt')
        if not self.assert_load_from_manifest(CD_manifest_file, self.tcga_cd_path):
            self.load_tcga_aml(CD_manifest_file, target_dir = self.tcga_cd_path)
        else : print('OUT TCGA + TARGET - AML Clinical Raw data already loaded locally on disk')   
        
        ## ASSEMBLE CLINICAL DATA ## 
        CD_tcga_profile = self.parse_clinical_xml_files()
        print ('OUT Assembled TCGA clinical data')
        # select target-aml most up to date clinical file 
        CD_target_filename = os.path.join(self.tcga_cd_path, os.listdir(self.tcga_cd_path)[-1])
        CD_target_profile = pd.read_excel(CD_target_filename)
        print ('OUT Assembled TARGET clinical data')
        pdb.set_trace()
        
        ## GET TARGET to CASE ID FILE ##
        filepath = os.path.join('Data', 'TCGATARGETAML', 'filename_to_caseID.csv')
        fileuuid_caseuuid = pd.read_csv(filepath) 
        
        ## MERGE TARGET + TCGA CLINICAL FEATURES ##
        tcga_target_clinical_features = merge_tcga_target_clinical_features(CD_tcga_profile, CD_target_profile)
        info_data = tcga_target_clinical_features.merge(fileuuid_caseuuid)
        # process filenames for easier fetching
        info_data['filepath'] = [os.path.join(GE_tcga_path,  filename_x) for filename_x in info_data.filename_x]
        # add a dataset tag
        info_data['dataset'] = 'C'
        info_data['sequencer'] = 'Hi-seq'
        # select proper columns
        info_data = info_data[['TARGET USI', 'submitter_id', 'filepath','dataset', 'sequencer', 'Gender', 'Risk group', 'FLT3/ITD positive?', 'NPM mutation','Induction_Type', 'Overall Survival Time in Days', 'Vital Status']]

        print("OUT Merged TCGA and TARGET clinical features")

        ## ASSEMBLE GE PROFILES FROM REPERTOIRE ##
        # unzip and merge files 
        import gzip
        # relabel repertoire column , store into matrix
        gene_repertoire['ensmbl_id'] = [g.split(".")[0] for g in gene_repertoire.gene_id]
        count_matrix = gene_repertoire[['ensmbl_id', 'Name', 'Category']]
        for i,r in info_data.iterrows() :
            if i % 10 == 0 : print('OUT Assembled TCGA + TARGET (C) {} / {} HT-Seq data'.format(i, info_data.shape[0] ))
            filename =  r['filepath']
            GE_profile = pd.read_csv(filename, sep = '\t', names = ['ensmbl_id', r['TARGET USI']])
            GE_profile['ensmbl_id'] = [e.split('.')[0] for e in GE_profile.ensmbl_id]
            count_matrix = count_matrix.merge(GE_profile, on = 'ensmbl_id') 
        print('OUT finished assembling TCGA + TARGET raw COUNT matrix of {} with {} samples'.format(count_matrix.shape[0], i + 1)) 
        return {'count_matrix': count_matrix, 'info_data' : info_data, 'gene_info': gene_repertoire} 
 
    def assert_load_from_manifest(self, manifest_file, tcga_path):
        """ 
        returns false if number of files in manifest and tcga_path unequal
        returns false if manifest doesnt exist
        """
        utils.assert_mkdir(tcga_path)
        manifest_exists = os.path.exists(manifest_file)
        tcga_manifest = len(os.listdir(tcga_path)) == len(open(manifest_file).readlines(  )) -1    
        return manifest_exists and tcga_manifest
        
    def parse_clinical_xml_files(self):
        # HARDCODED features to extract
        patient_features = ['batch_number', 'project_code', 'tumor_tissue_site', 'leukemia_specimen_cell_source_type', 'gender', 'vital_status','bcr_patient_barcode', 'days_to_death', 'days_to_last_known_alive', 'days_to_last_followup', 'days_to_initial_pathologic_diagnosis','days_to_birth', 'age_at_initial_pathologic_diagnosis', 'year_of_initial_pathologic_diagnosis' ]
        mutation_features = ['NPMc Positive', 'FLT3 Mutation Positive', 'Activating RAS Positive']
        header_array = np.concatenate((patient_features, mutation_features))
        clinical_data_matrix = []
        # assemble all .xml tcga-laml files
        for f in os.listdir(self.tcga_cd_path):
            if '.xml' in f:
                tree = ET.parse(os.path.join(self.tcga_cd_path, f))
                root = tree.getroot()
                # assemble a dict of all patient, clin features 
                xml_patient_dict = dict([(e.tag.split('}')[-1], e.text) for i in root for e in i])
                patient_features_array = [xml_patient_dict[f] for f in patient_features]
                mutation_features_array = []
                for e in root:
                    for i in e :
                        if i.tag.split('}')[-1] == "molecular_analysis_abnormality_testing_results":
                            mutation_profile = [(elem[0].text) for elem in i]
                            for mut in mutation_features:
                                mutation_features_array.append(int(mut in mutation_profile))	    
            clinical_data_matrix.append(np.concatenate((patient_features_array, mutation_features_array)))
        clinical_data = pd.DataFrame(clinical_data_matrix, columns = header_array)
        return clinical_data

    def merge_tcga_target_clinical_features(CD_tcga_profile, CD_target_profile):
        target_features = ['TARGET USI', 'Gender', 'FLT3/ITD positive?', 'NPM mutation', 'Overall Survival Time in Days', 'Vital Status', 'Risk group']
        target = CD_target_profile[target_features]
        tcga_features = ['bcr_patient_barcode','gender', 'FLT3 Mutation Positive', 'NPMc Positive', 'days_to_death', 'vital_status']
        tcga = CD_tcga_profile[tcga_features]
        tcga["Risk group"] = 'unknown'
        tcga.columns = target_features
        # uniformize values 
        target.Gender = target.Gender.str.upper()
        target['FLT3/ITD positive?'] = np.asarray(target['FLT3/ITD positive?'] == 'Yes', dtype = int)
        target['NPM mutation'] = np.asarray(target['NPM mutation'] == 'Yes', dtype = int)
        tcga_target_clinical_features = pd.DataFrame(np.concatenate((tcga, target)), columns = target_features)
        # add an unknown induction type column
        tcga_target_clinical_features['Induction_Type'] = 'unknown'
        # rename columns 
        return tcga_target_clinical_features 

    
    def assemble_merge_leucegene_GE(info_data, gene_repertoire):
        count_matrix = gene_repertoire[['ensmbl_id', 'Name', 'Category']]
        # add a dataset tag
        info_data['dataset'] = 'B'
        info_data['dataset'][info_data['RNASEQ_protocol'] == 'stranded'] = "A"
        info_data.sort_values(by = ['dataset'], inplace = True, ignore_index=True) 
        for i,r in info_data.iterrows() :
            if i % 10 == 0 : 
                print('OUT Assembled LEUCEGENE ({} - {})  {} / {} HT-Seq data'.format(r['dataset'], r['RNASEQ_protocol'] ,i, info_data.shape[0] ))
            filename =  os.path.join(r['star_path'], 'ReadsPerGene.out.tab')
            if os.path.exists(filename) : 
                GE_raw_count = pd.read_csv(filename, sep = '\t', names = ['gene_id', '+', '-', 'all'])
                GE_raw_count['ensmbl_id'] = [e.split('.')[0] for e in GE_raw_count.gene_id]
                GE_raw_count = GE_raw_count[['ensmbl_id', 'all']]
                GE_raw_count.columns = ['ensmbl_id', 'LGN-' + r['Genomic_Project_ID'] + '-' + r['sampleID']]
                GE_raw_count = GE_raw_count.groupby('ensmbl_id').sum()
                count_matrix = count_matrix.join(GE_raw_count, on = 'ensmbl_id', how = 'inner')  
            else: print ('{} doesnt exist !'.format(filename))
        print('OUT finished assembling LEUCEGENE raw COUNT matrix of {} with {} samples'.format(count_matrix.shape[0], i + 1)) 
        return count_matrix, gene_repertoire 

    def load_leucegene_data(gene_repertoire):
        # set a number of static variables
        data_path = os.path.join('Data/LEUCEGENE')
        
        ## LOAD and PROCESS CLINICAL features data ##
        info_path = os.path.join(data_path, 'API_getTable_86ad88.xls') # clinical features  
        features_list = ['Genomic_Project_ID', 'RNASEQ_protocol', 'Sex', 'Cytogenetic risk',  'FLT3-ITD mutation','NPM1 mutation', 'Induction_Type', 'Overall_Survival_Time_days', 'Overall_Survival_Status']
        info_data = pd.read_csv(info_path, sep = '\t').T
        info_data = process_info_data(info_data, features_list)
        ## ASSEMBLE and MERGE GENE EXPRESSIONS IN A COUNT MATRIX ##
        count_matrix, gene_info = assemble_merge_leucegene_GE(info_data, gene_repertoire)
        # exp_data, info_data , gene_info = load_data(exp_path, info_path)
        return {'count_matrix' : count_matrix, 'info_data': info_data, 'gene_info': gene_info}

    def assemble_merge_tcga_leucegene():
        gene_repertoire = process_gene_repertoire_data(pd.read_csv('Data/LEUCEGENE/gene_repertoire_1.csv', sep = '\t'))
        leucegene = load_leucegene_data(gene_repertoire) # datasets A(stranded), B(n-stranded)
        tcga = assemble_load_tcga_data(leucegene['gene_info']) # dataset C(n-stranded)
        assert_mkdir('SETS')
        leucegene['count_matrix'].to_csv('SETS/leucegene_AB_ge_counts.csv')
        leucegene['info_data'].to_csv('SETS/leucegene_AB_clinical_features.csv')
        tcga['count_matrix'].to_csv('SETS/tcga_C_ge_counts.csv')
        tcga['info_data'].to_csv('SETS/tcga_C_clinical_features.csv')
        return leucegene, tcga
    
    def load_tcga_aml(manifest_file, target_dir = "OUT"):
        # target directory
        assert_mkdir(target_dir)
        # import manifest
        manifest = pd.read_csv(manifest_file, sep = '\t')
        # cycle through filenames
        for i, row in manifest.iterrows():
            if not row['filename'] in os.listdir(target_dir) : 
                # store file_id
                file_id = row['id']
                # store data endpoint
                data_endpt = "https://api.gdc.cancer.gov/data/{}".format(file_id)
                # get response
                response = requests.get(data_endpt, headers = {"Content-Type": "application/json"})
                # The file name can be found in the header within the Content-Disposition key.
                response_head_cd = response.headers["Content-Disposition"]
                # store filename
                file_name = re.findall("filename=(.+)", response_head_cd)[0]
                output_file_name = os.path.join(target_dir, file_name)
                with open(output_file_name, "wb") as o:
                    o.write(response.content)
                    print('{} written to {}'.format( file_name, output_file_name))
            else : print ('{} Already in data base'.format(row['filename']))

class Leucegene_Dataset():
    def __init__(self, cohort, learning = True) -> None:
        self._init_CF_files()
        self.COHORT = cohort
        self.learning = learning # for machine learning data processing
        print(f"Loading ClinF {self.COHORT} file ...")
        self.CF_file = f"Data/lgn_{self.COHORT}_CF"
        self.CF = pd.read_csv(self.CF_file, index_col = 0)  # load in and preprocess Clinical Features file
        self.NS = self.CF.shape[0]
        print("Loading Gene Expression file ...")
        self._load_ge_tpm() # load in and preprocess Gene Expression file    
        self._set_data(rm_unexpr = True)     
    
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

    def _set_data(self, rm_unexpr = False):
          
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
        cds_data = Data(self.GE_CDS_LOG, self.CF, self.gene_info, name = f"{self.COHORT}_CDS", learning = self.learning)
        if rm_unexpr :  cds_data.remove_unexpressed_genes(verbose=1)
        # set TRSC data
        trsc_data = Data(self.GE_TRSC_LOG, self.CF, self.gene_info, name = f"{self.COHORT}_TRSC", learning = self.learning) 
        # set LSC17 data
        lsc17_data = Data(self.get_LSC17(), self.CF ,self.gene_info, name = f"{self.COHORT}_LSC17", learning = self.learning )
        FE_data =  Data(self.get_embedding(), self.CF, self.gene_info, name = f"{self.COHORT}_FE" , learning = self.learning) if self.EMB_FILE else None
        self.data = {"CDS": cds_data, "TRSC": trsc_data, "LSC17":lsc17_data, "FE": FE_data}
    
    def get_embedding(self):
        print("Fetching embedding file...")
        if self.EMB_FILE.split(".")[-1] == "npy":
            emb_x = np.load(self.EMB_FILE)
        elif self.EMB_FILE.split(".")[-1] == "csv":
            emb_x = pd.read_csv(self.EMB_FILE, index_col=0)
        x = pd.DataFrame(emb_x, index = self.GE_CDS_LOG.index)
        return x 

    def get_LSC17(self):
        
        lsc17 = pd.read_csv("Data/SIGNATURES/LSC17_expressions.csv", index_col = 0)
        #LSC17_expressions = self.x[self.x.columns[self.x.columns.isin(lsc17.merge(self.gene_info, left_on = "ensmbl_id_version", right_on = "featureID_x").featureID_y)]]
        return lsc17

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