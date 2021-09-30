# base
import pdb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', '.*do not.*', )

import experiments.parser as parser
args = parser.parse_arguments()

def main():
    # custom
    import experiments.experiments_Cox_1 as E
    #E.run_experiment_1(args)
    E.run_experiment_2(args)
if __name__ == "__main__":
    main()
