# Using *Mutual Interactors* to study the genetics of aging in *C. elegans*

*Bradley Buchner*

This repository is a detached fork of the [milieu](https://github.com/seyuboglu/milieu) repository that houses my personal implementation of the *Mutual Interactors* machine learning framework for predicting phenotype associations in the *C. elegans* genetic interaction network. Introduced by [Eyuboglu et al. (2023)](https://pubmed.ncbi.nlm.nih.gov/36540965/), *Mutual Interactors* is a network-based machine learning method that predicts gene-phenotype associations on the basis of a gene's shared neighbors (mutual interactors) with known phenotype-associated genes. For a more detailed description of the method, see [milieu/README.md](https://github.com/seyuboglu/milieu/blob/master/README.md) or [Eyuboglu et al. (2023)](https://pubmed.ncbi.nlm.nih.gov/36540965/). 

## Overview
Inspired by Eyuboglu et al. (2023), this repository employs *Mutual Interactors* to investigate a specific phenotype as a function of genetic interactions: the determination of an organism's lifespan (a.k.a. aging, longevity). I explore this in the notebook [milieu_worm_walkthrough.ipynb](notebooks/milieu_worm_walkthrough.ipynb) by training a model on the *C. elegans* interaction network, performing lifespan-associated node-set expansion to predict every node's (gene/protein's) association probability, and using the model's learned weights to quantify each node's influence as a mutual interactor in lifespan-associated node-set expansion, which, according to Eyuboglu et al. (2023), tends to be positively correlated with a node's involvement in cell-cell signaling and druggability. 

Once I train a model in [milieu_worm_walkthrough.ipynb](notebooks/milieu_worm_walkthrough.ipynb), I show that a *Mutual Interactors* model can infer phenotype-node associations in *C. elegans* significantly better than a randomized model. Specifically, the model is able to recover held-out nodes and reconstruct node sets at a high rate, scoring **30.1%** of held-out positives in the top 25 of predictions for phenotypes that the model did not see during training **(recall-at-25 = 30.1)**.

In [milieu_worm_walkthrough.ipynb](notebooks/milieu_worm_walkthrough.ipynb) I also demonstrate **how to quantify lifespan-specific mutual interactor influence**, which can be interpreted as a node's significance as a bridge connecting the lifespan-associated node-set. This score (which can be computed for any phenotype in the dataset) is calculated by integrating the model’s learned global interactor weights–which represent a node's capacity to carry phenotypic signal across the network—with its specific connectivity to known lifespan-associated nodes. More details on the score's derivation can be found in [milieu_worm_walkthrough.ipynb](notebooks/milieu_worm_walkthrough.ipynb).

Ultimately, this repository and its walkthrough Jupyter notebook serve as an example of how *Mutual Interactors* can be utilized to address more targeted research questions.

### Citation
Eyuboglu, S., Zitnik, M., & Leskovec, J. (2023). Mutual interactors as a principle for phenotype discovery in molecular interaction networks. Pacific Symposium on Biocomputing. Pacific Symposium on Biocomputing, 28, 61–72. https://pubmed.ncbi.nlm.nih.gov/36540965/
