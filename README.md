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
We recommend using a [conda environment](https://docs.conda.io/en/latest/miniconda.html)
for Python 3.7.
To setup the environment, run
```bash
conda env create --file environment.yml
# activate the environment
conda activate influence_theory
```
**Install Dependencies via Pip**:
To install dependencies, run
```bash
pip install -r requirement.txt
```
## Datasets
We used four different datasets, we outline them below. 
### Convex Experiements

__**Cash Transfer Experiement**__ 

This data is provided free by OPENICPSR and the American Economic Association. However, in order to download the data you must fill out a Terms of Use. Please download the file "table1.dta" by following the instrutions [here](https://www.openicpsr.org/openicpsr/project/113289/version/V1/view?path=/openicpsr/113289/fcr:versions/V1/table1.dta&type=file). 

Once downloaded, place the file under the folder "convex_exp"-->"data"-->"cash_transfer_data"

Citation: Angelucci, Manuela, and De Giorgi, Giacomo. Replication data for: Indirect Effects of an Aid Program: How Do Cash Transfers Affect Ineligibles’ Consumption?: table1.dta. Nashville, TN: American Economic Association [publisher], 2009. Ann Arbor, MI: Inter-university Consortium for Political and Social Research [distributor], 2019-10-12. https://doi.org/10.3886/E113289V1-130527

**Oregon Medicaid Experiment**

This data is provided free by the National Bureau of Economic Research. However, it also requires each user to acknowledge a Terms of Use. Once downloaded extract the following data files from the downloaded zip file "oregon_puf.zip".
  1. "OHIE_Public_Use_Files"--> "OHIE Data" --> "oregonhie_descriptive_vars.dta"
  2. "OHIE_Public_Use_Files"--> "OHIE Data" --> "oregonhie_survey12m_vars.dta"

Place both data files under the folder "convex_exp"-->"data"-->"oregon_data"

Citation: Oregon Health Insurance Experiment Web Page is available at www.nber.org/oregon

### Non-Convex Experiments (Language Model Experiements)
All data can be downloaded by from this [Google Drive]{https://drive.google.com/drive/folders/10O8SPuWVR-1YrHf0U8amYR90FGg7EayF?usp=sharing}. Place all data files (7 total) in the "data" folder in the main directory.

* Zero Shot Relations Extraction: We used a subset of the data used in [Mitchell et. al. 2022]{https://openreview.net/pdf?id=0DcZxeWfOPt}. The data was originally collected by [Levy et. al. 2017]{http://nlp.cs.washington.edu/zeroshot/} and split into train and test sets by [De Cao et al. (2021)]{https://arxiv.org/pdf/2104.08164.pdf}.
* WikiText: This is a subset of the huggingface WikiText103 by [Merity et. al. 2016]{https://arxiv.org/abs/1609.07843}
