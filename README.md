# Visualizations of the Leucegene dataset with dimensionality reduction

## 1. Introduction
In this report, we will investigate a subset of Gene Expression profiles coming from the Leucegene dataset. We will use both PCA, and t-SNE to perform dimensionality reduction on the data. This will provide visualizations of the data as well as highlighting putative cancer subgroups by eye. By correlating the most contributing genes to the PCA, we will assign each PC to a major ontology if it exists. 

## 2. Generating the Data

### 2.0 Initializing the program, setting up environment variables (taken from [Source](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) )

To install venv via pip
```{bash}
python3 -m pip install --user virtualenv
```

Then, create  activate the environment (Only first time)
```
python3 -m venv env
```

**Activate environment (everytime to run)**

**On windows**

do this before activating. (in a powershell)*
```
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser
```
Then, to activate the environment. One of the options.
```
./env/Scripts/Activate.ps1
./env/Scripts/activate
```

**On Unix**
```
source env/bin/activate
```

**Install required packages (Only first time)**
```
python3 -m pip install -r requirements.txt
```

Then finally, to run the program, run :
```{python3}
python3 main.py 
```
The other commands will be explained.

### 2.0.1: Experiment Book
## FIG1
```
# generate scores data, leave-one-out bootstraps c_index
python3 main.py --run_experiment 2 -C lgn_pronostic -P CF LSC17 PCA17 CF-PCA17 CF-LSC17 -M CPH -N_REP 10000 -CYT -O RES/FIGS/FIG4
```

## FIG2 
```
# generate log_reg GE to CF results (leave-one-out)
python3 main.py --run_experiment 2 -C lgn_pronostic -P PCA17 LSC17 PCA300 -M LOG_REG -O RES/FIGS/FIG2
```

## FIG3
```
## performance by dimension sweep (Leucegene)
python3 main.py --run_experiment 1 -C lgn_pronostic -P PCA CF-PCA RSelect RPgauss_var -IN_D 1 50 -N_REP 1000 -O RES/FIGS/FIG3 
## performance of LSC17
python3 main.py --run_experiment 1 -C lgn_pronostic -P LSC17 -IN_D 17 18 -N_REP 1000 -O RES/FIGS/FIG3
```

## FIG4
```
## performance by dimension sweep (Leucegene Intermediate)
python3 main.py --run_experiment 1 -C lgn_intermediate -P PCA -N_REP 1000 -IN_D 1 50 -O RES/FIGS/FIG4

## performance by dimension sweep (TCGA)
python3 main.py --run_experiment 1 -C TCGA -P PCA -N_REP 10000 -IN_D 1 5 -O RES/FIGS/FIG4

## performance of LSC17, PCAX (found precedently)
python3 main.py --run_experiment 2 -C lgn_intermediate -P PCAX LSC17 -M CPH -N_REP 10000 -O RES/FIGS/FIG4

```
### 2.1 Load data

* Loads in data
* Filter by CDS
* Performs TPM normalization

#### 2.1.1 Dumping outputfiles.
Allows faster retrieval of data tables over time and to also write the results it generated to disk. Output files, directories and filenames are generated automatically. 

* Writes every intermediate files during the loading sequences
* Writes results 

#### 2.1.2 Inspect base stats on data
***NOT YET IMPLEMENTED***
This will provide some basic stats of the data features a.k.a. the matrix of gene counts and the matrix of clinical features. 

```{python}
python3 main.py -INFO
with -O: outfiles
else: terminal
```

### 2.2 Datasets

### 2.1 Control over samples subset
```
name: cohort
flag: -C
type: str
values: ["public", "pronostic"]
default values: ["public", "pronostic"]
```
* "public" : The Leucegene public subset = 403 samples. Curated samples subset to complete different feature prediction on.  
* "pronostic" : The Leucegene pronostic subset = 300 samples. Curated samples subset that was selected to perform survival analysis on. 

### 2.2 Control over dimension of the features input space   
* TRSC: The Leucegene full transcriptome =  transcripts
* CDS: The Leucegene Coding transcriptome (22K coding sequences) 

The command:
```{python}
name: width
flag: -W
type: str
values: ["CDS", "TRSC"]
default values: ["CDS", "TRSC"]
```

### 2.3 PCA 
```{bash} 
flag: -PCA
```
Performs Principal Component Analysis on Leucegene Public, then writes output to file.
1. [Principal Components Analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis) (n_components = #samples)
2. **PCA Projection:** Performs gene count transformation on the PCs. This provides a dimensionality reduction if needed.  
3. Writes the loadings and the projection to file. 

```{python}
python3 main.py -PCA
default: OFF
```

### 2.4 Retrieving most contributing genes to Principal Components

We investigate the first 3 components and retrieve the most contributing genes with highest loadings values for each PC. Output is a table for each PC the 1000 most contributing genes. Then, gene set enrichment and significatn ontologies are retained. 
Following the methods described in [1](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0143196)

```{bash} 
flag: -GO
default: OFF
```
* Loads PCA loadings if they exist, else run PCA
* Correlates each gene contribution to PC with loadings
* Selects top 1000 genes for each PC 
* Performs Gene Ontology Enrichment analysis for each of the PC's top gene set
* Writes in table format 

```{python}
python3 main.py -PCA -GO
```

### 2.5 t-SNE on Leucegene Public with Clinical Features Annotations

#### 2.5.1 Running t-SNE (1 replicate, random perplexity)
* Dataset is shuffled (default)
* PCA reduction is applied (N PCs = N samples)
* Random perplexity is picked in range [15,30]
* Performs 2D t-SNE transformation on PCA of GE matrix

```{python}
python3 main.py -TSNE
default: OFF
```

#### 2.5.2 Running t-SNE with N replicates
* Random perplexity is picked within range [15, 30]
* For each replicate: 
    * dataset is shuffled 
    * a TSNE projection is produced

```{python}
python3 main.py -TSNE -N_TSNE <N>
default: 1
```
   
### 2.6 Plot Results 

* Loads in the PCA, t-SNE data if they exist, else performs PCA and T-SNE analyses. 
* Outputs a scatterplot BY **each feature** for  
    * PCA PC_X vs PC_Y | X,Y < 10 projections  
    * 2D t-SNE 

```{python}
python3 main.py -PCA -TSNE -PLOT
```

### 2.7 Automatic feature detection with Supervised Machine Learning 
pending...

## 3. Predicting Survival using Cox-PH and Derivates

## 3.1 Banchmarking
With this system we are training different models to predict survival and comparing their performance.

### 3.1.1 *Usage*
To specify which models to test, we enter the model types after the -B flag separated by spaces. The model types must be specified following the structure <model>-<input>, where the model is either a *Cox-Proportional Hazard, CPH-Deep neural netowrk*.  The input must be chosen between *LSC17, PCA or FE*. 

Note: When the CPH model type is used, by default the system will compute a CPHDNN1 network in parallel as well. This is to verify that our implementation of the CPH is able to compare to the *lifelines* implemetation that we use as baseline. 

```
python3 main.py -B FIXED_EMB <embedding_file>
name: BENCHMARKS
default: "CPH-PCA", "CoxSGD-PCA"
values: ["CPH-PCA", "CPH-LSC17", "CPH-FE", "CPHDNN-LSC17", "CPHDNN-PCA", "CPHDNN-FE", "CoxSGD-LSC17", "CoxSGD-PCA", "CoxSGD-FE"]
```
* CPH-PCA : Trains & test PCA + CPH model
* CPHDNN-PCA : Trains and test PCA + CPH-deep-neural-network model
* CPHDNN-FE : Trains & test  CPH-deep-neural-network model with a Factorized Embedding (fixed) as input.

#### 3.1.2 Double cross-validation (CV) system.
**Main Cross-validation:** In the current system, data is split into n folds, where n is set to 5 by default, so that every recorded performance is computed using a separate test sample, which has never been seen by the trained model. The goal of this system is to provide the least biased estimation possible of each of the data-model combinations accuracy on predicting risk on unseen data.

**Internal Cross-validation:** or ***Hyper-parameters optimisation*** of the model-data couplings. Whithin the training procedure for each model-data coupling, internal cross-validation searches for the optimal hyper-parameters that the main CV will evaluate. The model returned is the model that minimizes errors on its internal validation set, and this procedure is repeated across the whole training set. The number of optimisation steps for hyper-parameters is set with the -NOPT flag, set to 1 by default, then the number of internal cross-validation folds used for each step is set by the flag -INT_NFOLDS. 

The influence of these parameters NFOLDS, INT_NFOLDS is yet under study, but we hypothesize that they should not influence results greatly, but rather should reduce variane between technical replicates. On the other hand, putting high numbers of folds in each CV will increase computing time drastically. We suggest starting with low numbers of internal and main cross-val number to get rough estimates, and to increase this number to N (number of samples) afterwards as results get more and more stable.

***Rationale:*** We use cross-validation to search HP space because, by splitting the training set, we run the chance of under or over-perform on the validation set, just due to the randomness of the data-splitting action. This action may induce to have a large proportion of censored data, which is very uninfromative when testing a model. For an accurate estimate of the generalization power of each HP set tested, we choose to undergo full cross-validation, so that the recorded and compared accuracy represents the whole training set. The caveat of this method is that it pushes the system to choose a HP set that overfits the training set, which can lead to overfitting on the test set. 

```
python main.py -B <model1>-<input1> <model2>-<input1>, etc... -NFOLDS 5
name: NFOLDS, default: 5
name: INT_NFOLDs, default: 5
name: NOPT, default: 1
```

#### 3.1.3 training phase
* data is loaded, then a test set (10%) is reserved, rest is used to train models.
* training data is loaded into model, split into 5 folds 
* Internal cross-validation is performed 100 times and best HPs are stored.
* New model is created and trained with best HPs on whole training set.
* Optimized trained model is returned and ready to be tested on test set.
* repeat 10 times to get full cross-validated performance across cohort.

### 3.2.1

### 3.3 Simultaneous training of Fact-EMB-CPHDNN
```
python3 main.py -B CPHDNN-FE 
```
* Trains & test Factorized Embedding + CPHDNN model: *currently in progress*


## 5. Figures 

### 5.1 Benchmarks from 01-09-2021
