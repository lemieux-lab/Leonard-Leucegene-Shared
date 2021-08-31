# custom
import utils 
from engines import Engine
# base
import pdb

def main():
    args = utils.parse_arguments()
    eng = Engine(
        params = args, 
        )
    # load leucegene_public dataset
    eng.run() # for default CDS 
         
    if args.debug : pdb.set_trace()
    
if __name__ == "__main__":
    main()
