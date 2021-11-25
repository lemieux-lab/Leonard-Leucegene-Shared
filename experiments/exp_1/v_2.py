import pdb
import pandas as pd
import os
import numpy as np
from engines import utils
import datetime
def _merge_table_S():
    """ """
    print("OUT-Merging tables S (background) from experiment 3.1 ...")
    exp_dir = "RES/EXP_3.1"
    frames = []
    for table_path in os.listdir(exp_dir):
        inpath = os.path.join(exp_dir, table_path, "tableS.txt")
        frames.append(pd.read_csv(inpath))
    S = pd.concat(frames)
    S.index= np.arange(S.shape[0])
    return S

def run(args):
    # do appropriate imports    
    # load Background c_index distr tables 
    BG = _merge_table_S()
    outpath = utils.assert_mkdir("RES/TABLES/BACKGROUND")
    BG.to_csv(os.path.join(outpath, f"{datetime.datetime.now()}_background.csv"))

