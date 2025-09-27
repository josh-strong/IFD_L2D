# Identity Free Deferral

This project implements various models for training and evaluating IFD.

## Setup

To set up the environment, use the provided `environment.yml` file to create a conda environment:

```bash
conda env create -f environment.yml
conda activate ifd
```

## Datasets

The datasets used in the experiments are:
* HAM10000
* Blood Cells
* Liver Tumours (organs axial)

Blood Cells is automatically downloaded via medmnist. The rest must be manually downloaded, with instructions in the README.md files of the respective directories.

## Reproducing Main Experiments

You can reproduce the main experiments using the `run_exps.sh` script. This script will run the experiments for all datasets and seeds, and save the results in the `results` directory.

### Evaluation

To evaluate experiments, execute the following command:

```bash
python src/evaluate.py
```

## License

This code is provided solely and strictly for the purpose of reviewing the submission titled "Identity Free Deferral for Unseen Experts" for the ICLR 2025 conference. No other use, distribution, or modification of this code is permitted.

© 2025 CC anonymous authors. All rights reserved.