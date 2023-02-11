Our model called GNN-DOL enables mitosis detection by complementing a graph neural network
(GNN) with a differentiable optimization layer (DOL) that implements the constraint.

Data description

0. Installation

To run the GNN-DOL, you need to install the following dependencies:
torch-geometric   >=     1.7.1
cvxpy         >=         1.1.18
cvxpylayers    >=        0.1.5
scs             >=       3.2.0
hydra-core      >=       1.0.6
tqdm          >=         4.60.0
numpy        >=          1.18.5


1. Training
If you want to train the model, you need to run the following steps sequentially.

First, you need to generate the npy file for graph construction using

This file is used to constrcut graphs. Then, you need to run

to generate the graph structure file (.txt) and feature files for nodes and edges (.csv)



2. Prediction

3.Preprocessing