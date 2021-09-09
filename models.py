# imports
import pdb
from lifelines import CoxPHFitter
# classes 
class CPH():
    def __init__(self, data):
        self.wd = 1e-2
        self.data = data
    def _train(self):
        pdb.set_trace()
        CPH = CoxPHFitter(penalizer = self.wd, l1_ratio = 0.)
class CPHDNN():
    def __init__(self, data):
        self.data = data

def train(data, model_type, input):
    # define data
    data.set_input_targets(input)
    data.shuffle()
    if model_type == "CPH":
        model = CPH(data)
    elif model_type == "CPHDNN":
        model = CPHDNN(data)
    else: model = None
    model._train()

def main():
    # some test funcs
    pass

if __name__ == "__main__":
    main()