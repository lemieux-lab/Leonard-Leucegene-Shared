import engines.base_engines as eng

# base 
from datetime import datetime
import pdb 

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
    E.run_benchmarks()    
    if args.debug : pdb.set_trace()


