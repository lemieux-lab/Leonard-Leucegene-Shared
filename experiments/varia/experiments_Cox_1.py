import engines.base_engines as eng

# base 
from datetime import datetime
import pdb 
import pandas as pd

from engines.models.utils import assert_mkdir
def run_experiment_1(args):
    start = datetime.now()
    E = eng.Engine(
        params = args, 
    )
    E.run_visualisations() 
    end = datetime.now()
    print(f"DONE\t{end-start} s") 
    if args.debug : pdb.set_trace()

def run_experiment_2(args):
    E = eng.Engine(
        params = args, 
    )
    res = [E.run_benchmarks() for i in range(args.NREP_TECHN)]    
    tst_res = [res[i][0] for i in range(args.NREP_TECHN)]
    tr_res = [res[i][1] for i in range(args.NREP_TECHN)]
    tst_df = pd.concat(tst_res)
    tr_df = pd.concat(tr_res)
    res_table = tst_df.merge(tr_df, on = ["model_type", "repn_hash"])
    outpath = "RES/TABLES/BENCHMARKS"
    assert_mkdir(outpath)
    res_table.to_csv(f"{outpath}/{datetime.now()}.csv")
    pdb.set_trace()
    if args.debug : pdb.set_trace()

