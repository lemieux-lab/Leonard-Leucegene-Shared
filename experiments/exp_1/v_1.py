import pdb
import pandas as pd

def _produce_table_S(args):
    """Runs input models benchmark, outputs to dataframe"""
    print("OUT-Producing table S (benchmark)...")
    # create benchmarking engine
    from engines.base_engines import RP_BG_Engine
    b = RP_BG_Engine(params = args)
    outfile = b.run()
    data = pd.read_csv(outfile)
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
    