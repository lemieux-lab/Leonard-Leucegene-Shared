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
        from experiments.exp_1 import v_4 as exp_1_4
        from experiments.plots_1 import v_0 as plot_1_0
        self.experiment_dict = {
            "e1": exp_1_4,
            "e1.4": exp_1_4,
            "p1": plot_1_0
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
