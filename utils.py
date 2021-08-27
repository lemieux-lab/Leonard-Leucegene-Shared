import argparse 

def parse_arguments():
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument("-O", dest = "OUTFILES", action="store_true")
    parser.add_argument("-PCA", dest = "PCA", action="store_true")
    parser.add_argument("-TSNE", dest = "TSNE", action = "store_true")
    parser.add_argument("-PLOT", dest = "PLOT", action = "store_true")
    args = parser.parse_args()
    return args
