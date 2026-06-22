# Network intrusion detection on the UNSW-NB15 dataset using Self-Supervised approaches

This is the official implementation of the article *A Study on Self-Supervised Learning Objectives for Intrusion Detection*, for the 21th Conference of the Spanish Association for Artificial Intelligence (CAEPIA 2026). This repository also serves as an unnofficial implementation of the article *SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption*, using Pytorch Lightning.

The project performs contrastive pre-training to generate robust embeddings from tabular data. It evaluates the embedding's performance by transferring this knowledge to multiple supervised classifiers under a data-scarce scenario.

The framework is specifically configured to work with the UNSW-NB15 network intrusion dataset for multi-class classification (10 classes).


# Key features

- **Contrastive and Instance reconstruction pre-training**: Utilizes SCARF (and a instance reconstruction-based variant) via Pytorch Lightning to learn tabular representations without relying on labels.
- **Label scarcity simulation**: Simulates label scarcity by training downstream classifiers on a minimal fraction of labeled data (default 1%).
- **Robust multi-seed evaluation**: Automated execution across different seeds to compute stable means and 95% confidence intervals.
- **Model benchmarking**: Comparative evaluation of 6 classic and modern architectures stacked on top of the learned embeddings: Multi-Layer Perceptron, XGBoost, Random Forest, Support Vector Machine, K-Nearest Neighbors and C4.5.


# Repository structure

**UNSW-NB15**: Contains the UNSW-NB15 dataset in *parquet* format.

**SCARF.py**: Core SCARF model architecture definition.

**SCARFDataset.py**: Custom Pytorch Dataset for tabular data handling.

**SCARFEvaluation.py**: Take a trained SCARF model and test it on the test subset, training an MLP, RF and XGB model on top of the embeddings.

**fineTuner.py**: Take a trained SCARF model and fine-tune it for classification on the training set.

**featureExtractor.py**: Feature/embedding extractor from the frozen model.

**loss.py**: Definition of the contrastive loss.

**supervisedClassifier.py**: Downstream classification head.

**preprocessing.py**: Data loading and preprocessing logic.

**dataAnalysis**: Small script to analyze the UNSW-NB15 dataset.

**createGraphs**: Small script to create a bar graph to plot results.

**main.py**: Main execution script.


# Requirements and installation

This project is optimized to run on Python 3.10+ with the following dependencies:

``pip install torch lightning numpy scikit-learn xgboost pandas pyarrow``

The sript is configured to leverage hardware acceleration if CUDA drivers are properly installed.

# Usage

To run the entire pipeline (SCARF pre-training, feature extraction, downstream training for all six classifiers and multi-seed evaluation), execute:

``python main.py``

## Configurable parameters
Inside main.py, users can tune different parameters:

- ``seeds``: List of random seeds used to enforce statistical robustness and calculate confidence intervals.
- ``n_epochs``: Total epochs allocated for SCARF pre-training (default: 150).
- ``batch_size``: Number of batch created during SCARF pre-training (default: 256).
- ``doitsmall`` and ``label proportion``: Defines the downsamplling of the classification training set (default TRUE at 0.01 to simulate 1% of the training data).

Inside SCARF.py:

- ``self.ir_weight``: Define the weight of the reconstruction loss during training (default: 0.0)

# Results and metrics
Upon completion, the script automatically writes the evaluation report to a file named *results.txt*. This file logs hyperparameters configuration, per-class metrics (precision, recall, F1-score and AUC) across all ten target classes of UNSW-NB15, with 95% confidence intervals, and global performance metrics (overall accuracy, macro F1-score and weighted F1-score).