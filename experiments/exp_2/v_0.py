import pandas as pd
from engines.base_engines import Benchmark
from engines.datasets.base_datasets import SurvivalGEDataset
import os
import matplotlib.pyplot as plt

# the run command
def run(args):
    
    for cohort in args.COHORTS:
        print("Action 1: Loading data, preprocessing data...")
        SGE = SurvivalGEDataset()
        cohort_data = SGE.get_data(cohort)
        basepath = os.path.join("RES", f"EXP_{args.EXP}")
        print("done")
         
        # 30'
        outfile = os.path.join(basepath, f"summary_table_cf_{cohort}.csv" )
        print(f"Action 2: Printing summary data table of clinical infos. --> {outfile}")
        cohort_data.get_summary_table_cf().to_csv(outfile) 
        print(f"done") 
        pdb.set_trace()
        # 30'
        print(f"Action 3: Kaplan-Meier Curve of our survival data. --> {outfile}")
        outfile =os.path.join(basepath, f"kaplan_meier_{cohort}.csv" ) 
        ax = data.plot_kaplan_meier_curve()
        plt.savefig(outfile)
        print("done")

        # 45'
        print(f"Action 4: Corr PCA to clinical features. Corr LSC17 Express. to CF. {outfile}")
        outfile =os.path.join(basepath, f"corr_pca_lsc17_cf_{cohort}.csv" )   
        data.get_pca_lsc17_corr_to_cf().to_csv(outfile)
        print("done")

        # 60'
        # Clinical F only
        print("Action 5: Getting the performance of all CF features with CoxPH. ")
        outfile =os.path.join(basepath, f"benchmark_all_cf_cph_{cohort}.csv" ) 
        data.set_input(CF = data.get_binarized_clinical_features(), GE = None)
        cf_bm = GridBenchmark(data, args, grid)
        out = cf_bm.run(ndim = None)
        out.to_csv(outfile)
        print("done")

        # 30'
        # Gene Expression Only
        print("Action 6: Getting the performance of gene features with CoxPH. With different projection types.")
        in_D = 17
        outfile =os.path.join(basepath, f"benchmark_gene_exp_proj_{in_D}_{cohort}.csv" )
        data.set_input(CF = None, GE = data.get_gene_exp_features(in_D))
        ge_bm = GridBenchmark(data, args, grid)
        out = ge_bm.run(in_D)
        print("done")

        # 30'
        # Fully multivariate
        print("Action 7: Getting the performance of gene exp (17) + cf features")
        in_D = 17
        outfile =os.path.join(basepath, f"benchmark_gene_exp_proj_{in_D}_cf_{cohort}.csv" )
        data.set_input(CF = data.get_binarized_clinical_features(), GE = data.get_gene_exp_features(in_D))
        ge_cf_bm = GridBenchmark(data, args, grid)
        out = ge_cf_bm.run()
        print("done")

        # 60''
        # Gene Exp + single clinical feature type at a time
        
        # 60''
        # Gene Exp + all cf features - one type

        # 30'        
        # All cf Features + Gene Exp with increasing input dim

        # 120'' Figure Prep + # 60' Writing
        # 30' Prep 
        # 180' Cleanup + Pres 
        
        # total: 12h


