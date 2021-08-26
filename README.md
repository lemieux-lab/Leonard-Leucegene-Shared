# Visualizations of the Leucegene dataset with dimensionality reduction

## 1. Introduction
In this report, we will investigate a subset of Gene Expression profiles coming from the Leucegene dataset. We will use both PCA, and t-SNE to perform dimensionality reduction on the data. This will provide visualizations of the data as well as highlighting putative cancer subgroups by eye. By correlating the most contributing genes to the PCA, we will assign each PC to a major ontology if it exists. 

## 2. Generating the Data

### 2.1 Load data, and inspect the shape of data matrix

```{python}
python3 main.py
```

### 2.2 Separate data in 2
* The Leucegene full transcriptome (60K transcripts)
* The Leucegene Coding transcriptome (22K coding sequences) 

### 2.3 PCA on Leucegene Public
### 2.4 Retrieving most contributing genes to Principal Components
### 2.5 t-SNE on Leucegene Public with Clinical Features Annotations
### 2.6 Automatic feature detection with Supervised Machine Learning 

## 3. Discussion
### 3.1 Most contributing genes correlation to PC and GO enrichment
We investigate the first 3 components and retrieve the most contributing genes with highest correlation. Output is a table for each PC the 10 most contributing genes. Each column 

#### Table
PC# | GO enrichment
---|---
1 | x
2 | x
3 | x

### 3.2 t-SNE on 20K transcriptome with Clinical Features
