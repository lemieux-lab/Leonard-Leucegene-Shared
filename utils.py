import argparse 

def parse_arguments():
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("-O", dest = "OUTFILES", action="store_true")
    parser.add_argument("-TRSC", dest = "TRSC", action = "store_true", help = "run with full transcriptome")
    parser.add_argument("-PCA", dest = "PCA", action="store_true")
    parser.add_argument("-TSNE", dest = "TSNE", action = "store_true")
    parser.add_argument("-PLOT", dest = "PLOT", action = "store_true")
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
