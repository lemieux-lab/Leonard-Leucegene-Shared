# custom
import utils 
from datasets import Leucegene_Public_Dataset
from engines import Engine
# base
import pdb

def main():
    args = utils.parse_arguments()
    for w in args.WIDTHS:
        eng = Engine(
            params = args, 
            dataset = Leucegene_Public_Dataset(
                CF_file = "Data/lcg_public_CF",
                widths = w,
                cohort = "public"))
        # load leucegene_public dataset
        eng.run() # for default CDS 
         
    if args.debug : pdb.set_trace()
    
if __name__ == "__main__":
    main()
