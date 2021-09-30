# custom
import experiments.experiments_Cox_1 as CoxExp
import parser
# base
import pdb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', '.*do not.*', )

def main():
    args = parser.parse_arguments()
    coxExp.run_experiment_1(args)
    coxExp.run_experiment_2(args) 
if __name__ == "__main__":
    main()
