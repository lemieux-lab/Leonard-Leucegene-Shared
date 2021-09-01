# custom
import utils 
from engines import Engine
# base
import pdb
from datetime import datetime
def main():
    start = datetime.now()
    args = utils.parse_arguments()
    eng = Engine(
        params = args, 
        )
    # load leucegene_public dataset
    eng.run() # for default CDS 
    end = datetime.now()
    print(f"DONE\t{end-start} s")     
    if args.debug : pdb.set_trace()
    
if __name__ == "__main__":
    main()
