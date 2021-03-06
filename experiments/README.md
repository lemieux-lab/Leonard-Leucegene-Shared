# Experiment Book V1
## FIG1
```
# generate scores data, leave-one-out bootstraps c_index
python3 main.py --run_experiment 2 -C lgn_pronostic -P CYT CF LSC17 PCA17 CF-PCA17 CF-LSC17 -M CPH -N_REP 1e4
# plots c_index heatmap, km-fitting, survival curves
python3 main.py --run_experiment p1
```

## FIG2 
```
# generate log_reg GE to CF results (leave-one-out)
python3 main.py --run_experiment 2 -C lgn_pronostic -P PCA17 LSC17 PCA300 -M LOG_REG
# generate freq barplot, correlation heatmap, pc expl. var plot
python3 main.py --run_experiment p2 -C lgn_pronostic PCA30 
```

## FIG3
```
## performance by dimension sweep (Leucegene)
python3 main.py --run_experiment 1 -C lgn_pronostic -P PCA CF-PCA RSelect RPgauss_var -IN_D 1 50 -N_REP 1000
## performance of LSC17
python3 main.py --run_experiment 1 -C lgn_pronostic -P LSC17 -IN_D 17 18 -N_REP 1000
## plot results
python3 main.py --run_experiment p3
```

## FIG4
```
## performance by dimension sweep (Leucegene Intermediate)
python3 main.py --run_experiment 1 -C lgn_intermediate -P PCA -N_REP 1000 -IN_D 1 50
python3 main.py --run_experiment 1 -C lgn_intermediate -P LSC17 -N_REP 1000 -IN_D 17 18

## performance by dimension sweep (Leucegene Intermediate)
python3 main.py --run_experiment 1 -C tcga_target_aml -P PCA -N_REP 1000 -IN_D 1 50
python3 main.py --run_experiment 1 -C tcga_target_aml -P LSC17 -N_REP 1000 -IN_D 17 18

## plots
python main.py --run_experiment p1 -C lgn_pronostic_int -O RES/FIGS/FIG4
python main.py --run_experiment p1 -C tcga_target_aml -O RES/FIGS/FIG4

## performance of LSC17, PCAX (found precedently)
python3 main.py --run_experiment 2 -C lgn_intermediate -P PCA19 LSC17 -M CPH -N_REP 1000 -O RES/FIGS/FIG4
python3 main.py --run_experiment 2 -C tcga_target_aml -P PCA20 LSC17 -M CPH -N_REP 1000
-O RES/FIGS/FIG4
```
## Experiment Book V1.5

Dataset:
* Leucegene pronostic

Controls: 
* scoring by cytogenetic risk
* ridge-CPH from lifelines
* Cox-MLP from cox-nnet

Tested architectures:
* ridge-CPH own
* Cox-MLP own

Gene experession input:
* Full transcriptome input
* PCA* (with optimal nb PCs)
* LSC17

Regularisation:
* L2
* Dropout 

Output: 
* concordance index (bootsrapped 10K)
* survival curves (3 groups)

```
python3 main.py --run_experiment 3 -M ridge_cph_lifelines ridge_cph cphdnn cphdnn_coxnnet -P LSC17 full PCA -NDIMs 1 300 -O RES/RECOMB/FIG1 
```


### What is survival data and survival analysis?
Also called Time-to-event data, **survival data** presents recorded times before an event happened (eg. death). This data might be right-*censored*, meaning that we lost track of the sample before survey ended. It is a missing value that must be handled. Survival models try to associate a set of covariables called **X** to a survival function **S(t,X)**, dependent on time and the input covariables.

### What types of models can handle survival data with censorship? 
The Cox-Proportional-Hazard models, which come from 
* David R. Cox (1972), *Regression models and lifetables*
    Cox-Proportional-Hazard: FORMULA
Advantages of the Cox regression is that 1) it handles right-censored data. Then,because proportional hazard over time is assumed, 2) it provides a simple model fitted on data (linear) 3) and that does not depend on elapsed time to make prediction. 3) It is an interpretable system, in which we can retrieve the weights for downstream analysis. 

And their derivates, which implement deep-neural-networks to optimize the risk estimation function.
* Simons (1995), *A neural network for survival data*
    FORMULA
* Katzmann (2018), *Deepsurv: personalized treatment recommender system using a Cox proportional hazards deep neural network.*
    RESULT
* Lee (2018), *A Deep Learning Approach to Survival Analysis with Competing Risks*
    RESULT
* Kvamme (2019). *Time-to-Event Prediction with Neural Networks and Cox Regression.*
    RESULT


### Why don't we use time information as input to the models as well? 
In our context, we assume that we would use a risk scoring system as a prognosis score as early as possible, with little time between the cell extraction and sequencing (< 2 weeks), which would mean that time intervals would not have an exagerated impact on risk prediction.
In future work, we plan to test other networks that use time information as well, as it was shown to provide more accurate models like DeephHit (kvamme 2019, Lee 2018).


### Why do we need data representations in survival analysis? Why do we need to set up a specific approach for Gene Expression data?
Based on observations: 
* models train poorly with large input spaces generally. Because there are more parameters to optimize, optimization is harder and tends to overfit.
* models train well when using narrow input space. Because less parameters to train. Convergence is faster, less biased and more generalizable.
* Gene Expression data is very large, as the human genome contains about 20,000 coding sequences.
* With linear models such as the Cox-PH regression, if important interactions are present and need to be modelised, then modeling interactions from imput space increases the size of the input quadratically. (ex.: [x1,x2,x3,x4] -> [x1,x2,x3,x4,x1^2,x2^2,x3^2, x4^2, x1\*x2, x1\*x3, x1\*x4, x2\*x3, x2\*x4, x3\*x4]).
* Some representations of data may preserve survival dependency in the sample and reduce the size of the input. 


## Description of the approach
In the current work, instead of developping different Cox-PH adaptations, we propose to compare the performance of some of the above-presented algorithms with different data projections from various algorithms. We aim to demonstrate the ability of these algorithms to transform the input data into a form that helps training the survival networks.

With this work, we propose an alternative approach for the gene signature extraction in the context of survival. In this approach, we propose a system able to project the input features into a smaller embedding space, that retains survival dependencies and promotes good survival modelling via machine learning. 

The aim of this system is to provide a system that offers better accuracy than standard approaches. At the cost of interpretability, this system offers the advantage of not reliying on manual curation of specific gene sets, but rather to use the full Gene Expression array as input. This makes for a fast method for risk assessment in the context of cancer prognosis.

### What type of representations can we use to project the input features?
The methods we'll use to project the data fall into three major schemes: a filtering or selection scheme, a linear embedding scheme or a non-linear embedding scheme. 
```
python main.py -PCA -TSNE -C pronostic -W LSC17 -PLOT
```

### Which Cox model performs best on Leucegene data?
    What is a good performance?
        What is the right metric for computing survival prediction efficiency?
        Why do we use c index instead of log-rank test?
        What is the best way to evaluate generalization power of the models?

Is there something that makes Leucegene data reliable to other datasets.

What makes the LSC17 signature so strong in Leucegene?
How to control overfitting in non-linear complex models such as CPH-DNN?

### Extrapolation / perspectives.
What is the best representation-model coupling for survival from gene expression in any context?
What is the best representation-model coupling for risk prediction in AML.

## Experiments on projection algorithms
What is driving the PCA projection?
What is driving the projection with Factorized Embedding?
