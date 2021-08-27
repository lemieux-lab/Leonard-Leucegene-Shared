import argparse 

def parse_arguments():
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("-O", dest = "OUTFILES", action="store_true")
    parser.add_argument("-d", dest = "debug", action="store_true")
    # TRUE FALSE control parameters
    parser.add_argument("-PCA", dest = "PCA", action="store_true")
    parser.add_argument("-TSNE", dest = "TSNE", action = "store_true")
    parser.add_argument("-PLOT", dest = "PLOT", action = "store_true")
    # Addittive string parameters
    parser.add_argument("-W", dest = "WIDTHS", type = int, nargs = "+",  default = [1], help = "Dimensionality of input features space. \n1: Small transcriptome ~= 19,500 variables \n2: Full transcriptome ~= 54,500 transcripts. Can put as many jobs in this queue. Jobs will be done sequentially" )
    return args
    args = parser.parse_args()

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
