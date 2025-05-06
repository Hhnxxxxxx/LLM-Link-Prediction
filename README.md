# Untitled
Exploring citation link prediction with GCNs and comparing SciBERT vs. BoW embeddings.

## Overview

This repository contains all resources for our project on citation link prediction using graph neural networks. We construct a citation graph from real publication data, extract node embeddings using SciBERT and Bag-of-Words (BoW), and compare their performance using GCN-based models with different decoders. The repo includes the processed dataset, experimental scripts, trained models, and the final paper.

## File Structure

The repo is structured as:

- `Data` contains the data, including ...
- `Models` contains trained model checkpoints.
- `Models/tmp_BoW_dp.pt` contains a model trained using Bag-of-Words (BoW) node features and Dot Product decoder.
- `Models/tmp_SciBERT_mlp.pt` contains a model trained using SciBERT-based node embeddings and MLP decoder.
- `Other` contains some figure results used in the paper.
- `Paper` contains the PDF of the paper.
- `Scripts` contains the Python scripts used to preprocess data, train models, and evaluate link prediction performance.
