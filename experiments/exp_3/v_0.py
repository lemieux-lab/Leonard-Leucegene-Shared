from engines.datasets.base_datasets import SurvivalGEDataset
from engines.models import cox_models
import pdb 

def load_data(args):
    print("loading data")
    SGE = SurvivalGEDataset()
    cohort_data = SGE.get_data(args.COHORT)
    print("done")
    return cohort_data

def train_ridge_cph(train_data, alpha = 0):    
    model = cox_models.ridge_CPH(data=train_data)
    tr_metrics = model._train()
    pdb.set_trace()
    


def run(args):
    ## Takes input 
    cohort_data = load_data(args)
    # transforms it
    data = cohort_data["CDS"]
    data.split_train_test(5)
    train_data = data.folds[0].train
    test_data = data.folds[0].test
    # trains ridge cph
    train_c, train_l, model = train_ridge_cph(data, alpha = 0)
    # report training (c_index, loss)
    #plot_data(train_data)
    # tests
    #test_model(model, test_data)
    # reports (c_index, loss)  
    # print results  

