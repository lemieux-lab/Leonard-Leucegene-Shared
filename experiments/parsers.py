import argparse 

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_experiment", dest = "EXP", type = str, default = "3", help = "experiment number and version")
    # add control arguments
    parser.add_argument("-O", dest = "OUTFILES", action="store_true", help = "generate output files")
    parser.add_argument("-d", dest = "debug", action="store_true", help = "debug")
    parser.add_argument("-C", dest = "COHORTS", nargs = "+", type = str, default = ["public", "pronostic"], help = "public: \tThe Leucegene public subset = 403 samples. Curated samples subset to complete different feature prediction on.\n pronostic: The Leucegene pronostic subset = 300 samples. Curated samples subset that was selected to perform survival analysis on. ")
    parser.add_argument("-W", dest = "WIDTHS", nargs = "+", type = str,  default = ["CDS", "TRSC"], help = "Dimensionality of input features space. \n CDS: Small transcriptome ~= 19,500 variables \nTRSC: Full transcriptome ~= 54,500 transcripts. Can put as many jobs in this queue. Jobs will be done sequentially" )
    parser.add_argument("-N_TSNE", dest = "N_TSNE", type = int,  default = 1, help = "Number of T-SNE replicates done if TSNE selected. To check reproducibility." )
    parser.add_argument("-MAX_PC", dest = "MAX_PC", type = int,  default = 10, help = "Number of PC to be analysed (GO enrichment, figures)" )
    parser.add_argument("-GO_TOP_N", dest = "GO_TOP_N", type = int,  default = 1000, help = "Number of genes in gene set when performing GO enrichment analysis" )
    parser.add_argument("-FIXED_EMB", dest = "EMB_FILE", type = str, default = "Data/emb125_MLP25.csv", help = "name of embedding file used for fixed embedding mode CPH training prediction.")
    parser.add_argument("-M", dest = "MODEL_TYPES", type = str, nargs = "+", default = ["CPH"], help= "list of models to perform survival modelling.")
    parser.add_argument("-N_OPTIM", dest = "NREP_OPTIM", default = 1, type = int, help = "number of optimizations for Hyper parameters")
    parser.add_argument("-N_REP", dest = "NREP_TECHN", default = 10, type = int, help = "number of technical replicates") 
    parser.add_argument("-E", dest = "NEPOCHS", default = 1, type = int, help = "number of epochs for optim of DNN models")
    parser.add_argument("-IN_D", dest = "INPUT_DIMS", default = 17, type = int, help = " number of input dimensions to test in models")
    parser.add_argument("-P", dest = "PROJ_TYPES", type = str, nargs = "+", default = ["PCA", "SVD", "LSC17", "RPgauss", "RPsparse"], help= "list of projection types for survival prediction and further analyses")
    parser.add_argument("-P_BG", dest = "BG_PROJ_TYPE", type = str, default ="RPsparse", help= "projection type of background distribution of accuracies")
    
    # TRUE FALSE control parameters
    parser.add_argument("-PCA", dest = "PCA", action="store_true")
    parser.add_argument("-GO", dest = "GO", action = "store_true")
    parser.add_argument("-TSNE", dest = "TSNE", action = "store_true")
    parser.add_argument("-PLOT", dest = "PLOT", action = "store_true")
    parser.add_argument("-CPH", dest = "CPH", action = "store_true") 
    parser.add_argument("-CPHDNN", dest = "CPHDNN", action = "store_true")  
    
    ### FACTORIZED EMBEDDINGS ###
    ### Hyperparameter options ##
    parser.add_argument('--epoch', default=50, type=int, help='The number of epochs we want ot train the network.')
    parser.add_argument('--seed', default=123456, type=int, help='Seed for random initialization and stuff.')
    parser.add_argument('--batch-size', default=10000, type=int, help="The batch size.")
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    

    ### Dataset specific options
    parser.add_argument('--data-dir', default='Data/', help='The folder contaning the dataset.')
    parser.add_argument('--data-file', default='lgn_public_GE_CDS_TPM.csv', help='The data file with the dataset.')
    parser.add_argument('--dataset', choices=['gene', 'domaingene', 'impute', 'fedomains', 'doubleoutput'], default='gene', help='Which dataset to use.')
    parser.add_argument('--mask', type=int, default=0, help="percentage of masked values")
    parser.add_argument('--missing', type=int, default=0, help="number of held out combinations for FE domains")
    parser.add_argument('--data-domain', default='.', help='Number of domains in the data for triple factemb')
    parser.add_argument('--transform', default=True,help='log10(exp+1)')
    
    # Model specific options
    parser.add_argument('--layers-size', default=[50], type=int, nargs='+', help='Number of layers to use.')
    parser.add_argument('--emb_size', default=17, type=int, help='The size of the embeddings.')
    parser.add_argument('--set-gene-emb', default='.', help='Starting points for gene embeddings.')
    parser.add_argument('--warm_pca', default='.', help='Datafile to use as a PCA warm start for the sample embeddings')

    parser.add_argument('--weight-decay', default=1e-5, type=float, help='The size of the embeddings.')
    parser.add_argument('--model', choices=['factor', 'triple', 'multiple','doubleoutput', 'choybenchmark'], default='factor', help='Which model to use.')
    parser.add_argument('--cpu', action='store_true', help='If we want to run on cpu.') # TODO: should probably be cpu instead.
    parser.add_argument('--name', type=str, default=None, help="If we want to add a random str to the folder.")
    parser.add_argument('--gpu-selection', type=int, default=1, help="selectgpu")


    # Monitoring options
    parser.add_argument('--save-error', action='store_true', help='If we want to save the error for each tissue and each gene at every epoch.')
    parser.add_argument('--make-grid', default=True, type=bool,  help='If we want to generate fake patients on a meshgrid accross the patient embedding space')
    parser.add_argument('--nb-gridpoints', default=50, type=int, help='Number of points on each side of the meshgrid')
    parser.add_argument('--load-folder', help='The folder where to load and restart the training.')
    parser.add_argument('--save-dir', default='RES/FE_runs/', help='The folder where everything will be saved.')

    args = parser.parse_args()
    return args