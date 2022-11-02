# Run MIS experiements on zsRE data with Arnoldi approximation method
for alpha in 0.05 0.10
do
    python run_zsre_MIS.py results_dir="/results/zsre/MIS" n=49 device=cuda:2 data_seed=1 approx_method="arnoldi" model.pt="/models/zsre/model_zsre_1_49_200" method.arnoldi.n_it=30 method.arnoldi.top_k=10 method.arnoldi.lambda1=100 alpha=${alpha} +model=bart-base
    python run_zsre_MIS.py results_dir="/results/zsre/MIS" n=122 device=cuda:2 data_seed=1 approx_method="arnoldi" model.pt="/models/zsre/model_zsre_1_122_200" method.arnoldi.n_it=30 method.arnoldi.top_k=10 method.arnoldi.lambda1=100 alpha=${alpha}  +model=bart-base
    python run_zsre_MIS.py results_dir="/results/zsre/MIS" n=182 device=cuda:2 data_seed=1 approx_method="arnoldi" model.pt="/models/zsre/model_zsre_1_182_200" method.arnoldi.n_it=30 method.arnoldi.top_k=10 method.arnoldi.lambda1=100 alpha=${alpha}  +model=bart-base
    python run_zsre_MIS.py results_dir="/results/zsre/MIS" n=302 device=cuda:2 data_seed=1 approx_method="arnoldi" model.pt="/models/zsre/model_zsre_1_302_200" method.arnoldi.n_it=30 method.arnoldi.top_k=10 method.arnoldi.lambda1=100 alpha=${alpha}  +model=bart-base
    python run_zsre_MIS.py results_dir="/results/zsre/MIS" n=743 device=cuda:2 data_seed=1 approx_method="arnoldi" model.pt="/models/zsre/model_zsre_1_743_200" method.arnoldi.n_it=30 method.arnoldi.top_k=10 method.arnoldi.lambda1=100 alpha=${alpha}  +model=bart-base
    python run_zsre_MIS.py results_dir="/results/zsre/MIS" n=4499 device=cuda:2 data_seed=1 approx_method="arnoldi" model.pt="/models/zsre/model_zsre_1_4499_200"  method.arnoldi.n_it=30 method.arnoldi.top_k=10 method.arnoldi.lambda1=100 alpha=${alpha}  +model=bart-base
done
echo "Job completed at $(date)"