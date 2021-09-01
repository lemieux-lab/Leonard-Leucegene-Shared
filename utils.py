import argparse 

def parse_arguments():
    parser = argparse.ArgumentParser()
    # add control arguments
    parser.add_argument("-O", dest = "OUTFILES", action="store_true", help = "generate output files")
    parser.add_argument("-d", dest = "debug", action="store_true", help = "debug")
    parser.add_argument("-C", dest = "COHORTS", nargs = "+", type = str, default = ["public", "pronostic"], help = "public: \tThe Leucegene public subset = 403 samples. Curated samples subset to complete different feature prediction on.\n pronostic: The Leucegene pronostic subset = 300 samples. Curated samples subset that was selected to perform survival analysis on. ")
    parser.add_argument("-W", dest = "WIDTHS", nargs = "+", type = str,  default = ["CDS", "TRSC"], help = "Dimensionality of input features space. \n CDS: Small transcriptome ~= 19,500 variables \nTRSC: Full transcriptome ~= 54,500 transcripts. Can put as many jobs in this queue. Jobs will be done sequentially" )
    parser.add_argument("-N_TSNE", dest = "N_TSNE", type = int,  default = 1, help = "Number of T-SNE replicates done if TSNE selected. To check reproducibility." )
    
    # TRUE FALSE control parameters
    parser.add_argument("-PCA", dest = "PCA", action="store_true")
    parser.add_argument("-TSNE", dest = "TSNE", action = "store_true")
    parser.add_argument("-PLOT", dest = "PLOT", action = "store_true")
    # Addittive string parameters
    args = parser.parse_args()
    return args

# assert mkdir 
def assert_mkdir(path):
    import os
    """
    FUN that takes a path as input and checks if it exists, then if not, will recursively make the directories to complete the path
    """
        
    currdir = ''
    for dir in path.split('/'):
        dir = dir.replace('-','').replace(' ', '').replace('/', '_') 
        if not os.path.exists(os.path.join(currdir, dir)):
            os.mkdir(os.path.join(currdir, dir))
            print(os.path.join(currdir, dir), ' has been created')
        currdir = os.path.join(str(currdir), str(dir))
    return currdir
