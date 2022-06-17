import pdb
import pandas as pd
import numpy as np 
# import the Benchmarking engine
from engines.base_engines import Benchmark 
# import the SurvivalGEDatasets module
from engines.datasets.base_datasets import SurvivalGEDataset

def survey_in_d(data, args):
    # set input dim range
    input_dims = np.arange(args.INPUT_DIMS[0], args.INPUT_DIMS[1])
    # cycle through input dims
    for in_D in input_dims:    
        # init a Benchmarking Engine
        BM = Benchmark(data, args)
        # run the Benchmark and record metrics
        BM.run(in_D)

def run(args):
    
    # init the datasets module
    SGE = SurvivalGEDataset()
    # cycle through all datasets
    cohort = args.COHORT
    print(f"OUT-Producing benchmark table cohort: {cohort}...")
    # load cohort data
    data = SGE.get_data(cohort)
    # action 1
    survey_in_d(data, args)
        
        