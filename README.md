# Influence Theory Experiements
This repository contains the code and the scripts to reproduce the experiments 
[in this paper](NEED TO ADD LINK). 
In this paper we establish finish-sample statistical bounds under self-concordance, 
as well as computational complexity bounds for influence functions and 
approximate maximum influence perturbations using efficient inverse-Hessian-vector 
product implementations.

SHOULD I DESCRIBE INFLUENCE FUNCTIONS?? --> Influence functions are powerful statistical 
tools to identify influential datapoints. Computing the influence of a particular 
datapoint boils down to computing an inverse-Hessian-vector product.

## Dependencies
The code is written in Python and the dependencies are:
- Python >= 3.7.13
- PyTorch >= 1.1
- Huggingface Transformers >= 4.24.0
- scikit-learn >= 1.0.2

**Conda Environment**:
NEED HELP CREATING THIS....
We recommend using a [conda environment](https://docs.conda.io/en/latest/miniconda.html)
for Python 3.7.
To setup the environment, run
```bash
conda env create --file environment.yml
# activate the environment
conda activate mauve-experiments
```
In addition, you will have to install the following manually:
- PyTorch, version 1.7: [instructions](https://pytorch.org/get-started/locally/),
- HuggingFace Transformers, version 4.2.0: [instructions](https://huggingface.co/transformers).

The code is compatible with PyTorch >= 1.1.0 and transformers >= 3.2.0 but
we have not thoroughly tested it in this configuration.
