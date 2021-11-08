# base
import pdb
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', '.*do not.*', )

import experiments.parsers as parsers
args = parsers.parse_arguments()
#FE_args = parsers.parse_FE_arguments()
class ExperimentLoader:
    def __init__(self) -> None:
        from experiments import exp_3_0, exp_3_1, exp_3_2, exp_3_3
        self.experiment_dict = {
            "3.0": exp_3_0,
            "3.1": exp_3_1,
            "3.2": exp_3_2,
            "3.3": exp_3_3
        }
    def load_experiment(self, exp_vn):
        """Load and returns an Experiment Object"""
        return self.experiment_dict[exp_vn]

def main():
    # custom
    e = ExperimentLoader().load_experiment(args.EXP)
    e.run(args)
    # import experiments.experiments_FE as FE_exp
    # FE_exp.run_experiment(FE_args)
    #import experiments.experiments_PCA as PCAExperiments
    #PCAExperiments.experiment_1()
if __name__ == "__main__":
    main()
