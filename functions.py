from sklearn.decomposition import PCA 

def run_pca(dataset, outpath):
    print("Running PCA...")
    # init object
    pca = PCA()
    # fit to data
    pca.fit(dataset.GE_CDS_TPM_LOG)
    # get loadings
    # 
    # transform in self.NS dimensions
    # 
    # Writes to file 
    # 
    return pca  