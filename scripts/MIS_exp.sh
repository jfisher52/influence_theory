# Run MIS experiements on zsRE data with Arnoldi approximation method
n=(49 122 182 302 743 4499)

for alpha in 0.05 0.10
do
    python run_MIS_exp.py results_dir="/results/zsre/MIS" n=${n[0]} device=cuda:2 data_seed=1 approx_method="arnoldi" model.pt="/models/zsre/model_zsre_1_"${n[0]}"_200" method.num_epochs=30 method.arnoldi.top_k=10 method.regularization_param=100 alpha=${alpha} +model=bart-base 
    python run_MIS_exp.py results_dir="/results/zsre/MIS" n=${n[1]} device=cuda:2 data_seed=1 approx_method="arnoldi" model.pt="/models/zsre/model_zsre_1_"${n[1]}"_200" method.num_epochs=30 method.arnoldi.top_k=10 method.regularization_param=100 alpha=${alpha} +model=bart-base 
    python run_MIS_exp.py results_dir="/results/zsre/MIS" n=${n[2]} device=cuda:2 data_seed=1 approx_method="arnoldi" model.pt="/models/zsre/model_zsre_1_"${n[2]}"_200" method.num_epochs=30 method.arnoldi.top_k=10 method.regularization_param=100 alpha=${alpha} +model=bart-base 
    python run_MIS_exp.py results_dir="/results/zsre/MIS" n=${n[3]} device=cuda:2 data_seed=1 approx_method="arnoldi" model.pt="/models/zsre/model_zsre_1_"${n[3]}"_200" method.num_epochs=30 method.arnoldi.top_k=10 method.regularization_param=100 alpha=${alpha} +model=bart-base 
    python run_MIS_exp.py results_dir="/results/zsre/MIS" n=${n[4]} device=cuda:2 data_seed=1 approx_method="arnoldi" model.pt="/models/zsre/model_zsre_1_"${n[4]}"_200" method.num_epochs=30 method.arnoldi.top_k=10 method.regularization_param=100 alpha=${alpha} +model=bart-base 
    python run_MIS_exp.py results_dir="/results/zsre/MIS" n=${n[5]} device=cuda:2 data_seed=1 approx_method="arnoldi" model.pt="/models/zsre/model_zsre_1_"${n[5]}"_200" method.num_epochs=30 method.arnoldi.top_k=10 method.regularization_param=100 alpha=${alpha} +model=bart-base 
done
echo "Job completed at $(date)"