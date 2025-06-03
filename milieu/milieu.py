#!/usr/bin/env python
"""
Implementation of Milieu model as described in "Mutual Interactors as a principle for the 
discovery of phenotypes in molecular networks" by Sabri Eyuboglu, Marinka Zitnik, and
Jure Leskovec. 

Includes:

Milieu  a torch.nn.Module that implements a trainable Milieu model.
MilieuWorm  a torch.nn.Module that implements a trainable Milieu model
    modified for the discovery of phenotypes in the roundworm C. elegans.
MilieuDataset   a torch.util.data.Dataset that serves NodeSet expansion examples
MilieuDatasetWorm   a torch.util.data.Dataset that serves NodeSet expansion examples
    modified for the discovery of phenotypes in the roundworm C. elegans.

"""
import os
import json
import logging
from shutil import copyfile
from collections import defaultdict
from copy import deepcopy

import random
import numpy as np
import networkx as nx
import torch 
import torch.nn as nn 
from torch.autograd import Variable
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.sparse import coo_matrix, csr_matrix
import parse

from milieu.paper.methods.method import DPPMethod
from milieu.data.associations import load_node_sets
from milieu.util.metrics import compute_metrics
from milieu.util.util import set_logger, set_seed, load_mapping

__author__ = "Evan Sabri Eyuboglu"


class Milieu(nn.Module):
    """
    Milieu model as described in "Mutual Interactors as a principle for the 
    discovery of phenotypes in molecular networks".
    Milieu training is parameterized by the self.params dictionary. The default dictionary
    is updated with the params passed to __init__.
    """
    default_params = {
        "mps": True,
        "device": 0,

        "batch_size": 200,
        "num_workers": 4,

        "optim_class": "Adam",
        "optim_args": {
            "lr": 1e-1,
            "weight_decay": 0
        },

        "metric_configs": [
            {
                "name": "recall_at_25",
                "fn": "batch_recall_at",
                "args": {"k": 25}
            }
        ]
    }

    def __init__(self, network, params):
        """
        Milieu model as described in {TODO}.
        args:
            network (Network) The Network to use.
            params   (dict) Params to update in default_params.
        """
        super().__init__()

        set_logger()
        logging.info("Milieu")

        # override default parameters
        logging.info("Setting parameters...")
        self.params = deepcopy(self.default_params)
        self.params.update(params)

        self.network = network
        self.adj_matrix = network.adj_matrix


        logging.info("Building model...")
        self._build_model()
        logging.info("Building optimizer...")
        self._build_optimizer()
        logging.info("Done.")

    def _build_model(self):
        """
        Initialize the variables and parameters of the Milieu model. 
        See Methods, Equation (2) for corresponding mathematical definition. 
        """
        # degree vector, (D^{-0.5} in Equation (2))
        degree = np.sum(self.adj_matrix, axis=1, dtype=float)
        inv_sqrt_degree = np.power(degree, -0.5)
        inv_sqrt_degree = torch.tensor(inv_sqrt_degree, dtype=torch.float)

        # adjacency matrix of network, (A in Equation (2))
        adj_matrix = torch.tensor(self.adj_matrix, dtype=torch.float)

        # precompute the symmetric normalized adj matrix, used on the left of Equation (2)
        adj_matrix_left = torch.mul(torch.mul(inv_sqrt_degree.view(1, -1),
                                              adj_matrix),
                                    inv_sqrt_degree.view(-1, 1))

        # precompute the normalized adj matrix, used on the right of Equation (2)
        adj_matrix_right = torch.mul(inv_sqrt_degree.view(1, -1),
                                     adj_matrix)
        self.register_buffer("adj_matrix_right", adj_matrix_right)
        self.register_buffer("adj_matrix_left", adj_matrix_left)

        # milieu weight vector, ('W' in Equation (2))
        self.milieu_weights = nn.Parameter(torch.ones(1, 1, adj_matrix.shape[0],
                                                      dtype=torch.float,
                                                      requires_grad=True))

        # scaling parameter, ('a' in in Equation (2))
        self.scale = nn.Linear(1, 1)

        # the bias parameter, ('b' in Equation (2))
        self.bias = nn.Parameter(torch.ones(size=(1,),
                                            dtype=torch.float,
                                            requires_grad=True))

    def forward(self, inputs):
        """
        Forward pass through the model. See Methods, Equation (2).
        args:
            inputs (torch.Tensor) an (m, n) binary torch tensor where m = # of nodesets
            in batch and n = # of ndoes in the Network
        returns:
            out (torch.Tensor) an (m, n) torch tensor. Element (i, j) is the activations 
            for nodeset i and node j. Note: these are activations, not probabilities.
            Use torch.sigmoid to convert to probabilties. 
        """
        m, n = inputs.shape
        out = inputs  
        out = torch.matmul(inputs, self.adj_matrix_left)
        out = torch.mul(out, self.milieu_weights)
        out = torch.matmul(out, self.adj_matrix_right)

        out = out.view(1, m * n).t()
        out = self.scale(out) + self.bias
        out = out.view(m, n)

        return out

    def predict(self, inputs):
        """
        Make probabilistic predictions for expansions of a batch of 
        node sets.
        args:
            inputs (torch.Tensor) an (m, n) binary torch tensor where m = # of nodeset
            in batch and n = # of nodes in the PPI network
        returns:
            out (torch.Tensor) an (m, n) torch tensor. Element (i, j) is the probability 
            node j is associated with nodeset i and node j.
        """
        return torch.sigmoid(self.forward(inputs))
    
    def expand(self, node_names=None, top_k=10):
        """
        Get the top k nodes with the highest probability of association for a 
        phenotype of interest. Must provide nodes known to be associated with the 
        phenotype via either the entrez_ids or genbank_ids argument. 
        args:
            entrez_ids (iterable) a set of entrez ids representing nodes known to be 
            associated with a phenotype of interest. Note: must provide either entrez_ids
            or genbank ids, but not both. 

            genbank_ids (iterable) a set of genbank ids representing nodes known to be
            associated with a phenotype of interest. Note: must provide either entrez_ids 
            or genbank ids, but not both. 

            top_k (int) the number of predicted nodes to return
        returns:
            top_k_entrez/genbank    (list(tuple)) returns a list of tuples of the form
            (entrez/genbank_id, probability of association). Includes top k predictions
            with highest probability. 
        """
        # build model input vector 
        input_nodes = self.network.get_nodes(node_names)
        inputs = torch.zeros((1, len(self.network)), dtype=torch.float)
        inputs[0, input_nodes] = 1
        
        if self.params["mps"]:
            inputs = inputs.to(self.params["device"])

        probs = self.predict(inputs).cpu().detach().numpy().squeeze()

        # get top k predictions
        ranking = np.arange(len(self.network))[np.argsort(-probs)]
        top_k_nodes = []
        for node in ranking:
            if node not in input_nodes:
                top_k_nodes.append(node)
                if len(top_k_nodes) == top_k:
                    break

        top_k_probs = probs[top_k_nodes]
        top_k_entrez = list(self.network.get_names(top_k_nodes))

        return list(zip(top_k_entrez, top_k_probs))

    def loss(self, outputs, targets):
        """
        Compute weighted, binary cross-entropy loss as described in Methods, Equation (3).
        Positive examples are weighted {# of negative examples} / {# of positive examples}
        and negative examples are weighted 1. 
        args:
            outputs (torch.Tensor) An (m, n) torch tensor. Element (i, j) is the 
            activation for node set i and node j. Note: these are activations, not 
            probabilities. We use BCEWithLogitsLoss which combines the sigmoid with 
            the loss for numerical stability. 

            targets (torch.Tensor) An (m, n) binary tensor. Element (i, j) indicates
            whether node j is in the held-out set of nodes associated with node set
            i. 

        returns:
            out (torch.Tensor) A scalar loss.
        """
        num_pos = 1.0 * targets.data.sum()
        num_neg = targets.data.nelement() - num_pos
        bce_loss = nn.BCEWithLogitsLoss(pos_weight=num_neg / num_pos)
        return bce_loss(outputs, targets) 
    
    def train_model(self, train_dataset, valid_dataset=None): 
        """
        Train the Milieu model on train_dataset. Parameters for training including
        "num_epochs" and "optimizer_class" should be specified in the params dict 
        passed in at __init__.  Optionally validate the model on a validation dataset
        on each epoch. Computes metrics specified in params["metric_configs"] on each
        epoch. 
        args:
            train_dataset (MilieuDataset) A milieu dataset of training node sets
            valid_dataset (MilieuDataset) A milieu dataset of validation node sets
        returns:
            train_metrics   (list(dict)) train_metrics[i] is a dictionary mapping metric
            names to their values on epoch i
            valid_metrics   (list(dict)) like train_metrics but for validation metrics 
        """
        if "seed" in self.params:
            set_seed(self.params["seed"])
        logging.info(f'Starting training for {self.params["num_epochs"]} epoch(s)')

        # move to mps
        if self.params["mps"]:
            self.to(self.params["device"])

        train_metrics = []
        dl_generator = None
        if "seed" in self.params:
            dl_generator = torch.Generator()
            dl_generator.manual_seed(self.params["seed"])
        train_dataloader = DataLoader(train_dataset, 
                                      batch_size=self.params["batch_size"], 
                                      shuffle=True,
                                      generator = dl_generator,
                                      num_workers=self.params["num_workers"],
                                      # pin_memory=self.params["device"]
                                      )
        
        validate = valid_dataset is not None
        if validate:
            valid_metrics = []
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=self.params["batch_size"], 
                                          shuffle=True,
                                          generator=dl_generator,
                                          num_workers=self.params["num_workers"],
                                          # pin_memory=self.params["device"]
                                          )

        all_train_metrics = []
        for epoch in range(self.params["num_epochs"]):
            logging.info(f'Epoch {epoch + 1} of {self.params["num_epochs"]}')

            metrics = self._train_epoch(train_dataloader)
            train_metrics.append(metrics)

            if validate:
                metrics = self.score(valid_dataloader)
                valid_metrics.append(metrics)

        return train_metrics, valid_metrics if validate else train_metrics

    def _train_epoch(self, dataloader, metric_configs=[], verbose=False):
        """ Train the model for one epoch
        args:
            train_data  (DataLoader) A dataloader wrapping a MilieuDataset
            metric_configs  (list(dict)) A list of metric configuration dictionary. Each
            config dict should include "name", "fn", and "args". "fn" should be the name
            of a function in milieu.metrics. See default params for an example.
            verbose  (Boolean) Whether or no to output training updates
        return:
            metrics (dict)  Dictionary mapping metric "name" to value. 
        """
        logging.info("Training")

        self.train()

        metrics = defaultdict(list)

        avg_loss = 0

        with tqdm(total=len(dataloader)) as t:
            for i, (inputs, targets) in enumerate(dataloader):
                if self.params["mps"]:
                    inputs = inputs.to(self.params["device"])
                    targets = targets.to(self.params["device"])

                # forward pass
                outputs = self.forward(inputs)
                if verbose:
                    num_pos = 1.0 * targets.data.sum()
                    print(f"[train loss] positive examples = {num_pos}")
                loss = self.loss(outputs, targets)

                # backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss = loss.cpu().detach().numpy()
                # compute metrics
                probs = torch.sigmoid(outputs)
                compute_metrics(probs.cpu().detach().numpy(),
                                targets.cpu().detach().numpy(),
                                metrics,
                                self.params["metric_configs"])

                # compute average loss and update progress bar
                avg_loss = ((avg_loss * i) + loss) / (i + 1)
          
                t.set_postfix(loss='{:05.3f}'.format(float(avg_loss)))
                t.update()
                del loss, outputs, inputs, targets

        metrics = {name: np.mean(values) for name, values in metrics.items()}
        return metrics
    
    def score(self, dataloader, metric_configs=[]):
        """ Evaluate the model on the data in dataloader and the metrics in 
            metric_configs.
        args:
            train_data  (DataLoader) A dataloader wrapping a MilieuDataset
            metric_configs  (list(dict)) A list of metric configuration dictionary. Each
            config dict should include "name", "fn", and "args". See default params for
            example. 
        return:
            metrics (dict)  Dictionary mapping metric "name" to value. 
        """
        logging.info("Validation")
        self.eval()

        # move to mps
        if self.params["mps"]:
            self.to(self.params["device"])

        metrics = defaultdict(list)
        avg_loss = 0

        with tqdm(total=len(dataloader)) as t, torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                # move to GPU if available
                if self.params["mps"]:
                    inputs = inputs.to(self.params["device"])
                    targets = targets.to(self.params["device"])

                # forward pass
                probs = self.predict(inputs)
                compute_metrics(probs.cpu().detach().numpy(),
                                targets.cpu().detach().numpy(),
                                metrics,
                                self.params["metric_configs"])

                # compute average loss and update the progress bar
                t.update()

        return metrics
    
    def _build_optimizer(self,
                         optim_class="Adam", optim_args={}):
        """
        Build the optimizer. 
        args:
            optim_class (str) The name of an optimizer class from torch.optim
            optim_args (args) The args for the optimizer
        """
        optim_class = getattr(optim, self.params["optim_class"])
        self.optimizer = optim_class(self.parameters(), **self.params["optim_args"])

        # Build learning rate scheduler
        self.scheduler = None
        if self.params.get("lr_scheduler_class"):
            scheduler_class = getattr(torch.optim.lr_scheduler, self.params["lr_scheduler_class"])
            self.scheduler = scheduler_class(self.optimizer, **self.params["lr_scheduler_args"])

    def save_weights(self, destination):
        """
        Save the model weights. 
        args:
            destination (str)   path where to save weights
        """
        torch.save(self.state_dict(), destination)

    def load_weights(self, src_path):
        """
        Load model weights. 
        args:
            src_path (str) path to the weights files.
            substitution_res (list(tuple(str, str))) list of tuples like
                    (regex_pattern, replacement). re.sub is called on each key in the dict
        """
        if self.params["mps"]:
            src_state_dict = torch.load(src_path, 
                                        map_location=torch.device(self.params["device"]))
        else:
            src_state_dict = torch.load(src_path)

        self.load_state_dict(src_state_dict, strict=False)
        n_loaded_params = len(set(self.state_dict().keys()) & set(src_state_dict.keys()))
        n_tot_params = len(src_state_dict.keys())
        if n_loaded_params < n_tot_params:
            logging.info("Could not load these parameters due to name mismatch: " +
                         f"{set(src_state_dict.keys()) - set(self.state_dict().keys())}")
        logging.info(f"Loaded {n_loaded_params}/{n_tot_params} pretrained parameters " +
                     f"from {src_path}.")

class MilieuWorm(Milieu):
    """
    Milieu model as described in "Mutual Interactors as a principle for the
    discovery of phenotypes in molecular networks" adapted for C. elegans aging study.
    Milieu training is parameterized by the self.params dictionary. The default dictionary
    is updated with the params passed to __init__.
    """
    default_params = {
        "mps": True,
        "device": 0,

        "batch_size": 1,
        "num_workers": 0,
        "num_epochs": 200,
        "early_stopping_patience": 500,
        "edge_dropout_alpha": 0.5,

        "optim_class": "Adam",
        "optim_args": {
            "lr": 1e-1,
            "weight_decay": 0
        },

        "metric_configs": [
            {
                "name": "recall_at_25",
                "fn": "batch_recall_at",
                "args": {"k": 25}
            }
        ]
    }

    def __init__(self, network, params, use_weighted_adj=False, edge_dropout=False):
        """
        Milieu model as described in "Mutual Interactors as a principle for the
        discovery of phenotypes in molecular networks" adapted for C. elegans
        aging study.
        args:
            network (Network) The Network to use.
            params   (dict) Params to update in default_params.
            use_weighted_adj (Boolean) Use weighted adjacency matrix to build model
            edge_dropout (Boolean) Use edge dropout during model training
        """
        self.use_weighted_adj = use_weighted_adj

        # # rescale W to have mean = 1
        # W = network.weighted_adj_matrix.copy()
        # mean_w = W[W > 0].mean()
        # self.weighted_adj_matrix = W / mean_w

        self.weighted_adj_matrix = network.weighted_adj_matrix

        self.edge_dropout = edge_dropout

        self.alpha = params.get("edge_dropout_alpha", 1.0)

        super().__init__(network, params)

        # assume network.edge_confidence_matrix is your matrix W_norm with values in [0,1]
        self.register_buffer("edge_confidence_matrix", torch.tensor(self.weighted_adj_matrix, dtype=torch.float))
        self.register_buffer('adj_matrix_tensor', torch.tensor(self.adj_matrix, dtype=torch.float))
        self.register_buffer('identity', torch.eye(self.adj_matrix_tensor.size(0)))


    def _build_model(self):
        """
        Initialize the variables and parameters of the Milieu model.
        See Methods, Equation (2) for corresponding mathematical definition.
        """

        adj_matrix = self.weighted_adj_matrix if self.use_weighted_adj else self.adj_matrix

        # degree vector, (D^{-0.5} in Equation (2))
        degree = np.sum(adj_matrix, axis=1, dtype=float)
        degree = np.maximum(degree, 1e-6)  # <-- no value below 1e-6
        inv_sqrt_degree = np.power(degree, -0.5)
        if np.any(np.isinf(inv_sqrt_degree)) or np.any(np.isnan(inv_sqrt_degree)):
            print("inv_sqrt_degree has bad entries!",
                  inv_sqrt_degree[np.where(np.isinf(inv_sqrt_degree))],
                  inv_sqrt_degree[np.where(np.isnan(inv_sqrt_degree))])

        inv_sqrt_degree = torch.tensor(inv_sqrt_degree, dtype=torch.float)

        # adjacency matrix of network, (A in Equation (2))
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.float)

        # precompute the symmetric normalized adj matrix, used on the left of Equation (2)
        adj_matrix_left = torch.mul(torch.mul(inv_sqrt_degree.view(1, -1),
                                              adj_matrix),
                                    inv_sqrt_degree.view(-1, 1))

        # precompute the normalized adj matrix, used on the right of Equation (2)
        adj_matrix_right = torch.mul(inv_sqrt_degree.view(1, -1),
                                     adj_matrix)
        self.register_buffer("adj_matrix_right", adj_matrix_right)
        self.register_buffer("adj_matrix_left", adj_matrix_left)

        # milieu weight vector, ('W' in Equation (2))
        self.milieu_weights = nn.Parameter(torch.ones(1, 1, adj_matrix.shape[0],
                                                      dtype=torch.float,
                                                      requires_grad=True))

        # scaling parameter, ('a' in in Equation (2))
        self.scale = nn.Linear(1, 1)

        # the bias parameter, ('b' in Equation (2))
        self.bias = nn.Parameter(torch.ones(size=(1,),
                                            dtype=torch.float,
                                            requires_grad=True))

    def forward(self, inputs):
        """
        Forward pass through the model. See Methods, Equation (2).
        args:
            inputs (torch.Tensor) an (m, n) binary torch tensor where m = # of nodesets
            in batch and n = # of ndoes in the Network
        returns:
            out (torch.Tensor) an (m, n) torch tensor. Element (i, j) is the activations
            for nodeset i and node j. Note: these are activations, not probabilities.
            Use torch.sigmoid to convert to probabilties.
        """
        # adjacency matrix of network, (A in Equation (2))
        # adj_matrix = torch.tensor(self.adj_matrix, dtype=torch.float, device=inputs.device)

        # n = adj_matrix.size(0)
        # identity = torch.eye(n, device=inputs.device, dtype=torch.float)

        # drop edges if random.random() < q:
        q = random.random()
        if self.edge_dropout and self.training and q < 0.5:
            # sample mask; same shape as adjacency
            p = (self.alpha + (1-self.alpha)*self.edge_confidence_matrix)
            p = p.clamp(0.0, 1.0)
            mask = torch.bernoulli(p)
            adj_matrix_masked = (self.adj_matrix_tensor * mask) + self.identity

            # degree vector, (D^{-0.5} in Equation (2))
            # degree = np.sum(adj_matrix_masked, axis=1, dtype=float)
            degree = adj_matrix_masked.sum(dim=1)
            # inv_sqrt_degree = np.power(degree, -0.5)
            # inv_sqrt_degree = torch.tensor(inv_sqrt_degree, dtype=torch.float)
            inv_sqrt_degree = degree.pow(-0.5)

            adj_matrix_left = torch.mul(torch.mul(inv_sqrt_degree.view(1, -1),
                                                  adj_matrix_masked),
                                        inv_sqrt_degree.view(-1, 1))

            adj_matrix_right = torch.mul(inv_sqrt_degree.view(1, -1),
                                         adj_matrix_masked)
        else:
            adj_matrix_left, adj_matrix_right = self.adj_matrix_left, self.adj_matrix_right

        m, n = inputs.shape
        out = inputs
        out = torch.matmul(inputs, adj_matrix_left)
        out = torch.mul(out, self.milieu_weights)
        out = torch.matmul(out, adj_matrix_right)

        out = out.view(1, m * n).t()
        out = self.scale(out) + self.bias
        out = out.view(m, n)

        return out

    def loss(self, outputs, targets, targets_weight_mask):
        """
        Compute weighted, binary cross-entropy loss as described in Methods, Equation (3).
        Positive examples are weighted {# of negative examples} / {# of positive examples}
        and negative examples are weighted 1.
        args:
            outputs (torch.Tensor) An (m, n) torch tensor. Element (i, j) is the
            activation for node set i and node j. Note: these are activations, not
            probabilities. We use BCEWithLogitsLoss which combines the sigmoid with
            the loss for numerical stability.

            targets (torch.Tensor) An (m, n) binary tensor. Element (i, j) indicates
            whether node j is in the held-out set of nodes associated with node set
            i.

            targets_weight_mask (torch.Tensor) An (m, n) float tensor. Element (i, j)
            indicates the weight of node j in node set i, reflecting the confidence
            of its phenotypic association. The mask nudges the model to be better at
            recovering high-confidence positives by up-weighting their influence in
            the loss function.

        returns:
            out (torch.Tensor) A scalar loss.
        """
        bce_loss = nn.BCEWithLogitsLoss(weight=targets_weight_mask)
        return bce_loss(outputs, targets)

    def train_model(self, train_dataset, valid_dataset=None):
        """
        Train the Milieu model on train_dataset. Parameters for training including
        "num_epochs" and "optimizer_class" should be specified in the params dict
        passed in at __init__.  Optionally validate the model on a validation dataset
        on each epoch. Computes metrics specified in params["metric_configs"] on each
        epoch.
        args:
            train_dataset (MilieuDataset) A milieu dataset of training node sets
            valid_dataset (MilieuDataset) A milieu dataset of validation node sets
        returns:
            train_metrics   (list(dict)) train_metrics[i] is a dictionary mapping metric
            names to their values on epoch i
            valid_metrics   (list(dict)) like train_metrics but for validation metrics
        """
        if "seed" in self.params:
            set_seed(self.params["seed"])
        logging.info(f'Starting training for {self.params["num_epochs"]} epoch(s)')

        # move to mps
        if self.params["mps"]:
            self.to(self.params["device"])

        dl_generator = None
        if "seed" in self.params:
            dl_generator = torch.Generator()
            dl_generator.manual_seed(self.params["seed"])

        train_metrics = []
        train_losses = []
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=self.params["batch_size"],
                                      shuffle=True,
                                      generator=dl_generator,
                                      num_workers=self.params["num_workers"],
                                      # pin_memory=self.params["device"]
                                      )

        validate = valid_dataset is not None
        if validate:
            valid_metrics = []
            valid_losses = []
            valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=self.params["batch_size"],
                                          shuffle=False,
                                          generator=dl_generator,
                                          num_workers=self.params["num_workers"],
                                          # pin_memory=self.params["device"]
                                          )
            best_val_loss = float('inf')
            patience = self.params.get("early_stopping_patience", 10)  # Default patience of 10 epochs
            trigger_times = 0
            best_model_state = None

        for epoch in range(self.params["num_epochs"]):
            logging.info(f'Epoch {epoch + 1} of {self.params["num_epochs"]}')

            epoch_metrics, epoch_loss = self._train_epoch(train_dataloader)
            train_metrics.append(epoch_metrics)
            train_losses.append(epoch_loss)

            if self.scheduler:
                self.scheduler.step()

            if validate:
                epoch_metrics_val, epoch_loss_val = self.score(valid_dataloader)
                valid_metrics.append(epoch_metrics_val)
                valid_losses.append(epoch_loss_val)

                if epoch_loss_val < best_val_loss:
                    best_val_loss = epoch_loss_val
                    trigger_times = 0
                    best_model_state = deepcopy(self.state_dict())
                else:
                    trigger_times += 1
                    if trigger_times >= patience:
                        logging.info(
                            f'Early stopping triggered at epoch {epoch + 1} due to no improvement in validation loss for {patience} epochs.')
                        self.load_state_dict(best_model_state)  # Load the best model weights
                        break
            else:
                best_model_state = deepcopy(self.state_dict())  # Save the last model if no validation

        if validate:
            return train_metrics, valid_metrics, train_losses, valid_losses
        else:
            return train_metrics, train_losses

    def _train_epoch(self, dataloader, metric_configs=[], verbose=False):
        """ Train the model for one epoch
        args:
            train_data  (DataLoader) A dataloader wrapping a MilieuDataset
            metric_configs  (list(dict)) A list of metric configuration dictionary. Each
            config dict should include "name", "fn", and "args". "fn" should be the name
            of a function in milieu.metrics. See default params for an example.
            verbose  (Boolean) Whether or no to output training updates
        return:
            metrics (dict)  Dictionary mapping metric "name" to value.
        """

        self.train()

        metrics = defaultdict(list)

        avg_loss = 0

        for i, (inputs, targets, targets_weight_mask) in enumerate(dataloader):
            if self.params["mps"]:
                inputs = inputs.to(self.params["device"])
                targets = targets.to(self.params["device"])
                targets_weight_mask = targets_weight_mask.to(self.params["device"])

            # forward pass
            outputs = self.forward(inputs)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("⚠️Found NaN/Inf in activations!")
            if verbose:
                num_pos = 1.0 * targets.data.sum()
                print(f"[train loss] positive examples = {num_pos}")
            loss = self.loss(outputs, targets, targets_weight_mask)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss = loss.cpu().detach().numpy()
            # compute metrics
            probs = torch.sigmoid(outputs)
            compute_metrics(probs.cpu().detach().numpy(),
                            targets.cpu().detach().numpy(),
                            metrics,
                            self.params["metric_configs"])

            # compute average loss and update progress bar
            avg_loss = ((avg_loss * i) + loss) / (i + 1)

            del loss, outputs, inputs, targets, targets_weight_mask

        metrics = {name: np.mean(values) for name, values in metrics.items()}
        return metrics, avg_loss

    def score(self, dataloader, metric_configs=[], verbose=False):
        """ Evaluate the model on the data in dataloader and the metrics in
            metric_configs.
        args:
            train_data  (DataLoader) A dataloader wrapping a MilieuDataset
            metric_configs  (list(dict)) A list of metric configuration dictionary. Each
            config dict should include "name", "fn", and "args". See default params for
            example.
            verbose  (Boolean) Whether or no to output training updates
        return:
            metrics (dict)  Dictionary mapping metric "name" to value.
        """
        # logging.info("Validation")
        self.eval()

        # move to mps
        if self.params["mps"]:
            self.to(self.params["device"])

        metrics = defaultdict(list)
        avg_loss = 0

        # with tqdm(total=len(dataloader)) as t, torch.no_grad():
        with torch.no_grad():
            for i, (inputs, targets, targets_weight_mask) in enumerate(dataloader):
                # move to GPU if available
                if self.params["mps"]:
                    inputs = inputs.to(self.params["device"])
                    targets = targets.to(self.params["device"])
                    targets_weight_mask = targets_weight_mask.to(self.params["device"])

                # forward pass
                outputs = self.forward(inputs)
                if verbose:
                    num_pos = 1.0 * targets.data.sum()
                    print(f"[val loss] positive examples = {num_pos}")
                loss = self.loss(outputs, targets, targets_weight_mask)  # Calculate loss for validation as well
                loss = loss.cpu().detach().numpy()
                probs = self.predict(inputs)
                compute_metrics(probs.cpu().detach().numpy(),
                                targets.cpu().detach().numpy(),
                                metrics,
                                self.params["metric_configs"])

                # compute average loss and update the progress bar
                avg_loss = ((avg_loss * i) + loss) / (i + 1)
            logging.info(f"avg_val_loss: {round(float(avg_loss), 5)}")

        return metrics, avg_loss

class MilieuDataset(Dataset):

    def __init__(self, network, node_sets=None, frac_known=0.9):
        """
        PyTorch dataset that holds node sets and serves them to the
        Milieu for training. During training we simulate node set expansion by
        splitting each node set into an input set and a target set. Each time we
        access an node set set from this dataset we randomly sample 90% of associations
        for the input set and use the remaining 10% for the target set. See Methods.
        args:
            network (Network)    Network being used by the Milieu model
            node_sets    (list(NodeSet)) list of milieu.data.associations.NodeSet.
            frac_known  (float)   fraction of each association set used for input set and
            target set.
        """
        self.n = len(network)
        self.examples = [{"id": node_set.id,
                          "nodes": node_set.to_node_array(network)}
                         for node_set
                         in node_sets]
        self.frac_known = frac_known

    def __len__(self):
        """ Returns the size of the dataset."""
        return len(self.examples)

    def get_ids(self):
        """ Get the set of all the node_set ids in the dataset."""
        return set([node_set["id"] for node_set in self.examples])

    def __getitem__(self, idx):
        """
        Get an association split into an input and target set as described in Methods.
        args:
            idx (int) The index of the association set in the dataset.
        returns:
            inputs (torch.Tensor) an (n,) binary torch tensor indicating the nodes in
            the input set.
            targets (torch.Tensor) an (n,) binary torch tensor indicating the nodes in
            the target set.
        """
        nodes = self.examples[idx]["nodes"]
        np.random.shuffle(nodes)
        split = int(self.frac_known * len(nodes))

        known_nodes = nodes[:split]
        hidden_nodes = nodes[split:]

        inputs = torch.zeros(self.n, dtype=torch.float)
        inputs[known_nodes] = 1
        targets = torch.zeros(self.n, dtype=torch.float)
        targets[hidden_nodes] = 1

        # ensure no data leakage
        assert (torch.dot(inputs, targets) == 0)

        return inputs, targets

class MilieuDatasetWorm(Dataset):

    def __init__(self, network, phenotype_confidence_dict=None,
                 node_sets=None, frac_known=0.9):
        """
        PyTorch dataset that holds node sets and serves them to the 
        Milieu for training. During training we simulate node set expansion by 
        splitting each node set into an input set and a target set. Each time we
        access a node set from this dataset we randomly sample 90% of associations
        for the input set and use the remaining 10% for the target set. See Methods. 

        Args:
            network (Network): Network being used by the Milieu model
            phenotype_confidence_dict (dict): Dictionary of confidence scores
            for each node's phenotypic association
            node_sets (list(NodeSet)): list of milieu.data.associations.NodeSet.
            frac_known (float): fraction of each association set used for input set and
            target set.
        """
        # self.node_mapping = node_mapping
        self.n = len(network)
        self.examples = [{"id": node_set.id, 
                          "nodes": node_set.to_node_array(network)}
                         for node_set 
                         in node_sets]
        self.frac_known = frac_known
        self.phenotype_confidence_dict = phenotype_confidence_dict
    
    def __len__(self):
        """ Returns the size of the dataset."""
        return len(self.examples)
    
    def get_ids(self):
        """ Get the set of all the node_set ids in the dataset."""
        return set([node_set["id"] for node_set in self.examples])

    def __getitem__(self, idx):
        """
        Get an association split into an input and target set as described in Methods. 
        args:
            idx (int) The index of the association set in the dataset. 
        returns:
            inputs (torch.Tensor) an (n,) binary torch tensor indicating the nodes in
            the input set.
            targets (torch.Tensor) an (n,) binary torch tensor indicating the nodes in
            the target set.
        """
        nodes = self.examples[idx]["nodes"]
        np.random.shuffle(nodes)
        split = int(self.frac_known * len(nodes))

        known_nodes = nodes[:split]
        hidden_nodes = nodes[split:]

        inputs = torch.zeros(self.n, dtype=torch.float)
        targets = torch.zeros(self.n, dtype=torch.float)
        targets_weight_mask = torch.ones(self.n, dtype=torch.float)

        inputs[known_nodes] = 1
        targets[hidden_nodes] = 1

        num_pos = 1.0 * targets.data.sum()
        num_neg = targets.data.nelement() - num_pos
        pos_imbalance = num_neg / num_pos

        # if self.phenotype_confidence_dict is not None:
        #     for u in known_nodes:
        #         inputs[u] = self.phenotype_confidence_dict.get(self.node_mapping[str(u)], 0.0)
        #     for u in hidden_nodes:
        #         c = self.phenotype_confidence_dict.get(self.node_mapping[str(u)], 0.0)
        #         targets_weight_mask[u] = 1 + (pos_imbalance - 1) * c
        # else:
        #     for u in hidden_nodes:
        #         targets_weight_mask[u] = pos_imbalance
        for u in hidden_nodes:
            targets_weight_mask[u] = pos_imbalance

        # ensure no data leakage
        assert(torch.dot(inputs, targets) == 0)

        return inputs, targets, targets_weight_mask

