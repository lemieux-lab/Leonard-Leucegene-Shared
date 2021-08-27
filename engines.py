from datetime import datetime
import utils 
import functions

class Engine:
    def __init__(self, params, dataset):
        self.OUTFILES = params.OUTFILES # bool
        self.RUN_PCA = params.PCA # bool
        self.RUN_TSNE = params.TSNE # bool
        self.RUN_PLOTTING = params.PLOT # bool
        self.COHORT = "public" # HARDCODE
        self.dataset = dataset # Leucegene_Public_Dataset

        self.OUTPATHS = {   # dict
            "RES": utils.assert_mkdir(f"RES{datetime.now()}"),
        }


    def run(self):
        mode = {"params":1,"default":2}[mode]
        # outputs info
        self.dataset.dump_infos(self.OUTPATHS["RES"])
        # computes pca
        if self.RUN_PCA : 
            self.PCA = functions.run_pca(self.dataset)
            if self.OUTFILES : 
                print(self.PCA) #### write to file !!
            # writes table 
        # computes tsne
        if self.RUN_TSNE or mode :
            self.GE_TSNE, self.TSNE = functions.run_tsne(self.dataset)
            # writes table
            if self.OUTFILES or mode:
                print(self.TSNE)
            # scatter plot 
            if self.RUN_PLOTTING or mode:
                functions.run_tsne_plotting(self.dataset,self.GE_TSNE,  self.TSNE, self.OUTPATHS["RES"], cohort = self.COHORT, protein_coding = self.dataset.CDS, mode=mode)
                # by features
                # by PC 1...10 x PC 1...10
