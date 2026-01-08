# Applying the *Mutual Interactors* framework to study the genetics of aging in *C. elegans*

*Bradley Buchner*

This repository is a detached fork of the [milieu](https://github.com/seyuboglu/milieu) repository that repurposes its *Mutual Interactors* framework for the discovery of phenotype associations in the *C. elegans* genetic interaction network. For a detailed description of the method, see [milieu/README.md](https://github.com/seyuboglu/milieu/blob/master/README.md) or [Eyuboglu et al. (2023)](https://pubmed.ncbi.nlm.nih.gov/36540965/). 

Inspired by Eyuboglu et al. (2023), this repository extends the *Mutual Interactors* framework to investigate how an organism's genetic interactions determine its lifespan. The notebook `milieu_worm_walkthrough.ipynb` explores this by training a model on the *C. elegans* interaction network, performing lifespan-associated node-set expansion to predict every node's (gene/protein's) association probability, and using the model's learned weights to quantify each node's influence as a mutual interactor in lifespan-associated node-set expansion, which, according to Eyuboglu et al. (2023), is positively correlated with a node's involvement in cell-cell signaling and its druggability. 

Once trained, `milieu_worm_walkthrough.ipynb` shows that a *Mutual Interactors* model can infer phenotype-node associations in *C. elegans* significantly better than a randomized model. Specifically, the model is able to recover held-out nodes and reconstruct node sets at a high rate, scoring **30.1%** of held-out positives in the top 25 of predictions for phenotypes that the model did not see during training **(recall-at-25 = 30.1)**.

`milieu_worm_walkthrough.ipynb` also shows the process of calculating lifespan-specific mutual interactor influence, which can be interpreted as a node's significance as a bridge connecting the lifespan-associated node-set. This score is computed by integrating the model’s learned global interactor weights–which represent a node's capacity to carry phenotypic signal across the network—with its specific connectivity to known lifespan-associated nodes. By identifying nodes that act as significant mutual interactors for the aging phenotype, the framework can uncover potential candidates for perturbation. More details on the score's derivation are in `milieu_worm_walkthrough.ipynb`.

Ultimately, this repository serves as an example of how the *Mutual Interactors* framework can be extended to address more targeted research questions.
