# base
import pdb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', '.*do not.*', )

import experiments.parsers as parsers
Cox_args = parsers.parse_Cox_arguments()
#FE_args = parsers.parse_FE_arguments()

def main():
    # custom
    #import experiments.experiments_Cox_1 as CoxExperiments
    # E.run_experiment_1(args)
    #CoxExperiments.run_experiment_2(Cox_args)
    # import experiments.experiments_FE as FE_exp
    # FE_exp.run_experiment(FE_args)
    import experiments.experiments_PCA as PCAExperiments
    PCAExperiments.experiment_1()
if __name__ == "__main__":
    main()
