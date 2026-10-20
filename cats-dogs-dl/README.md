# Cats and Dogs++ — multi-label pet classification

- **Task:** [Kaggle competition](https://www.kaggle.com/competitions/cats-and-dogs/overview). Predict five labels per photo (`Cat`, `Dog` and three specific pets `Moris`, `Motya`, `Biatrix`); the metric is macro F1. Best single model: EfficientNet-B0, public LB 0.863.
- **How to run:** open the notebook on Kaggle, add the competition data via *Add Input*, upload `folds.csv`, enable *GPU T4* and *Internet*, add a `WANDB_API_KEY` secret, then *Run All*.
- **Stack:** PyTorch, PyTorch Lightning, timm, scikit-learn, Weights & Biases (no Keras / TensorFlow / Chainer).
- **Experiments:** all runs are logged in the public [W&B project](https://wandb.ai/lyaroshenko-kse/cats-and-dogs-pp); the summary table is in `results.csv`.
- **Report:** [`report.md`](report.md) covers EDA, metric, validation, experiments, diagnostics, ensemble and conclusions; figures are in `images/`.
