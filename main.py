# custom
import utils 
from datasets import Leucegene_Public_Dataset
from engines import Engine
# base
import pdb

def main():
    eng = Engine(
        args = utils.parse_arguments(), 
        dataset = Leucegene_Public_Dataset(
            CF_file = "Data/lcg_public_CF"))
    # load leucegene_public dataset
    eng.run()
    pdb.set_trace()
    
if __name__ == "__main__":
    main()
