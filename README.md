# Influence Theory Experiments
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
### Convex Experiments

**1. Cash Transfer Experiment**

This data is provided free by OPEN ICPSR and the American Economic Association. However, in order to download the data you must fill out a Terms of Use. Please download the file "table1.dta" by following the instructions [here](https://www.openicpsr.org/openicpsr/project/113289/version/V1/view?path=/openicpsr/113289/fcr:versions/V1/table1.dta&type=file). 

Once downloaded, place the file under the folder `convex_exp\data\cash_transfer_data` 

Citation: Angelucci, Manuela, and De Giorgi, Giacomo. Replication data for: Indirect Effects of an Aid Program: How Do Cash Transfers Affect Ineligibles’ Consumption?: table1.dta. Nashville, TN: American Economic Association [publisher], 2009. Ann Arbor, MI: Inter-university Consortium for Political and Social Research [distributor], 2019-10-12. https://doi.org/10.3886/E113289V1-130527

**2. Oregon Medicaid Experiment**

This data is provided free by the National Bureau of Economic Research. However, it also requires each user to acknowledge a Terms of Use. Please download the zip file "oregon_puf" by following the instructions [here](https://www.nber.org/research/data/oregon-health-insurance-experiment-data). Once downloaded extract the following data files from the downloaded zip file "oregon_puf.zip".
  1. "OHIE_Public_Use_Files"--> `OHIE_Data\oregonhie_descriptive_vars.dta`
  2. "OHIE_Public_Use_Files"--> `OHIE_Data\oregonhie_survey12m_vars.dta`

Place both data files under the folder `convex_exp\data\oregon_data` 

Citation: Oregon Health Insurance Experiment Web Page is available at www.nber.org/oregon

### Non-Convex Experiments (Language Model Experiments)
All the data for the two non-convext (language model) experiments can be downloaded from this [Google Drive](https://drive.google.com/drive/u/2/folders/10O8SPuWVR-1YrHf0U8amYR90FGg7EayF). Place all data files (7 total) in the `data` folder in the main directory. Below are details of this data.

**3. Zero Shot Relations Extraction (zsRE)**

We used a subset of the data used in [Mitchell et. al. 2022]{https://openreview.net/pdf?id=0DcZxeWfOPt}. The data was originally collected by [Levy et. al. 2017]{http://nlp.cs.washington.edu/zeroshot/} and further edited by [De Cao et al. (2021)]{https://arxiv.org/pdf/2104.08164.pdf}.

**4. WikiText**

This is a subset of the huggingface WikiText103 by [Merity et. al. 2016]{https://arxiv.org/abs/1609.07843}

## Experimental Pipeline (Convex Experiments)
Experimental code for both of the simulations (linear and logistic regression) and both of the economic experiments (Oregon Medicaid and Cash Transfer) can be found in the `convex_exp` folder. Each experiment is outlined in a juypter notebook and can be run by using the "Run All" function. Below is the folder location for each experiment.
* Linear Regression: `convex exp\simulation\simulation_linear_exp.pdf`
* Logistic Regression: `convex exp\simulation\simulation_logistic_exp.pdf`
* Oregon Medicaid: `convex exp\economic_exp\oregon_medicaid.pdf`
* Cash Transfer: `convex exp\economic_exp\cash_transfer.pdf`

## Experimental Pipeline (Non-Convex Experiments)
For each dataset (zsRE and WikiText), the experimental pipeline is as follows:
1. finetune the pretrained models on a range of subsets of the total data
2. compute the influence on a single test point using each of the four approximation method
3. compute the Most Influential Subset for five pre-selected test points using the Arnoldi method

The creation of the finetune models (Step 1) must be run first. Other steps can proceed in any order.

Here is how to find the scripts step-by-step for the zsRE experiement. Repeat the below steps for WikiText by replacing "zsre" with "wiki". 

**Step 1. Finetune the Models:**
Create the folder `models/zsre` in the base directory. Then, run `scripts/train_model_zsre` to generate six models of Bart-base finetuned on a subset of the original data. 

This script further finetunes a Bart-base (or distil GPT2 for WikiText) on a subset of training data n. All outputs will appear as `models/zsre/model_zsre_${data_seed}_${n}_${n_test}`.

This should take about 1.5 hours for the zsRE experiment and ???? hours for WikiText experiment. 

**Step 2. Approximated Influence of Single Point:**
Create the folder `results/zsre` in the base directory. Then, run `scripts/IF_exp_zsre` to generate the approximated influence of a training point for five different points using conjugate gradients, SGD, SVRG, and the Arnoldi method.

This script calculates the approximated influence of a specific training point on test set loss. We run this for 5 different training points over each of the four approximation methods outlined in the paper (conjugate gradients, SGD, SVRG, Arnoldi). All outputs will appear as `results/zsre/results_{config.task}_{config.n}_{config.approx_method}_{config.method.num_epochs}_{config.regularization_param}.pt"`.

This should take about ???? hours for the zsRE experiment and ???? hours for WikiText experiment. 

**Step 3. Approximated Most Influential Subset:**
Create the folder `results/zsre/MIS` in the base directory. Then, run `scripts/MIS_exp_zsre` to generate the approximated most influential subsets for five pre-selected test points, found using the Arnoldi method.

This script calculates the both the Most Influential Subset and the approximated influence of the Most Influencial Subset on the test set loss, at differing values of $\alpha$. We run this for 5 pre-selected test points and use the Arnoldi approximation. The Most Influential Subsets output to `results/zsre/results_MIS_{config.task}_{config.n}_{config.method.arnoldi.n_it}_{config.alpha}.pt"` and the approximated influence outputs to `results/zsre/results_MISinfluence_{config.task}_{config.n}_{config.method.num_epoch}_{config.alpha}.pt"`

This should take about ??? hours for the zsRE experiment and ???? hours for WikiText experiment. 

## Citation
If you find this repository useful, or you use it in your research, please cite:
```
NEED TO ADD
```
    
## Acknowledgements
This work was supported by the Institute for Foundations of Data Science. ASK ZAID ABOUT OTHERS!!!!!
