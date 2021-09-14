# custom
import utils 
from engines import Engine
# base
import pdb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', '.*do not.*', )
def main():
    start = datetime.now()
    args = utils.parse_arguments()
    eng = Engine(
        params = args, 
        )
    eng.run_visualisations() 
    end = datetime.now()
    print(f"DONE\t{end-start} s") 
    eng.run_benchmarks()    
    if args.debug : pdb.set_trace()
    
if __name__ == "__main__":
    main()
