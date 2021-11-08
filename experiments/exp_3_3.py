import pdb
import pandas as pd

def _produce_table_S(args):
    """
    -ASSERT Data is shuffled 
    -RUN Random Background 
    R = 1000
    [M -> [1,2,5,10,50,100,200,500,1000], RDM_PROJ[R], RDM_SIGN[R]] 
    -RUN PCA
    -RUN LSC17  
    outputs to dataframe"""
    print("OUT-Producing table S (benchmark)...")
    # run the random projection engine
    from engines.base_engines import RP_BG_Engine # Random Propjection Background Engine
    rp_bg = RP_BG_Engine(params = args)
    outfile = rp_bg.run()
    # run the benchmark engine 
    from engines.base_engines import Benchmark
    b = Benchmark(params = args)
    b.run()
    pdb.set_trace()


def _produce_tSNE_B(args):
    """Outputs tSNE of given data inputs in array of plots."""
    print("OUT-Producing TSNEs ...")

def _produce_table_D(args):
    """Computes sample distances in input and given projections. Outputs to DF"""

def run(args):
    # do appropriate imports    
    # produce table S
    S = _produce_table_S(args)