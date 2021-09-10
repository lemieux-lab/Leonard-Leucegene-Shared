# imports
import pdb
from lifelines import CoxPHFitter
import pandas as pd
import functions

# classes 
class CPH():
    def __init__(self, data):
        self.wd = 1e-2
        self.data = data

    def _train(self):
        # create lifelines dataset
        ds = pd.DataFrame(self.data.train.x.iloc[:,:25])
        ds["T"] = self.data.y["t"]
        ds["E"] = self.data.y["e"]
        CPH = CoxPHFitter(penalizer = self.wd, l1_ratio = 0.)
        self.model = CPH.fit(ds, duration_col = "T", event_col = "E")
        l = self.model.log_likelihood_
        c = self.model.concordance_index_
    
    def _test(self):
        self.out = self.model.predict_log_partial_hazard(self.data.test.x)
        return self.out 

class CPHDNN():
    def __init__(self, data):
        self.data = data

def train_test(data, model_type, input):
    # define data
    data.set_input_targets(input)
    data.shuffle()
    data.split_train_test(0.2)
    if model_type == "CPH":
        model = CPH(data)
    elif model_type == "CPHDNN":
        model = CPHDNN(data)
    else: model = None
    model._train()
    out = model._test()
    c_index = functions.compute_c_index(data.test.y["t"], data.test.y["e"], out)
    print (f"C index for model {model_type}, input: {input}: {c_index}")
    pdb.set_trace()

def main():
    # some test funcs
    pass

if __name__ == "__main__":
    main()