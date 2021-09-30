import pandas as pd
import numpy as np
import pdb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os 
import umap

import engines.base_engines as eng

def run_experiment(FE_args):
    E = eng.FE_Engine(
        params = FE_args,
    )
    E.run_fact_emb()
    if FE_args.debug : pdb.set_trace()

