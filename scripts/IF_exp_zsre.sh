# Run all zsRE approxmiation of influence function experiements
# Set parameters
lr=(5e-4 5e-4 5e-4 5e-4)
num_epochs=(100 50 25 30)
method=(conjugate_gradient sgd svrg arnoldi identity)
n=(49 122 182 302 743 4499)

# Run experiements
for i in 0 1 2 3
do
    python run_IF_exp.py results_dir="/results/zsre" task='zsre' iterations=5 tr_bsz=4 te_bsz=32 n=${n[0]} n_test=200 device=cuda:2 data_seed=1 approx_method=${method[i]} model.pt="/models/zsre/model_zsre_1_49_200" method.regularization_param=100  method.lr=${lr[i]} method.num_epochs=${num_epochs[i]} +model=bart-base
    python run_IF_exp.py results_dir="/results/zsre" task='zsre' iterations=5 tr_bsz=4 te_bsz=32 n=${n[1]} n_test=200 device=cuda:2 data_seed=1 approx_method=${method[i]} model.pt="/models/zsre/model_zsre_1_122_200" method.regularization_param=100  method.lr=${lr[i]} method.num_epochs=${num_epochs[i]} +model=bart-base 
    python run_IF_exp.py results_dir="/results/zsre" task='zsre' iterations=5 tr_bsz=4 te_bsz=32 n=${n[2]} n_test=200 device=cuda:2 data_seed=1 approx_method=${method[i]} model.pt="/models/zsre/model_zsre_1_182_200" method.regularization_param=100  method.lr=${lr[i]} method.num_epochs=${num_epochs[i]} +model=bart-base
    python run_IF_exp.py results_dir="/results/zsre" task='zsre' iterations=5 tr_bsz=4 te_bsz=32 n=${n[3]} n_test=200 device=cuda:2 data_seed=1 approx_method=${method[i]} model.pt="/models/zsre/model_zsre_1_302_200" method.regularization_param=100  method.lr=${lr[i]} method.num_epochs=${num_epochs[i]} +model=bart-base
    python run_IF_exp.py results_dir="/results/zsre" task='zsre' iterations=5 tr_bsz=4 te_bsz=32 n=${n[4]} n_test=200 device=cuda:2 data_seed=1 approx_method=${method[i]} model.pt="/models/zsre/model_zsre_1_743_200" method.regularization_param=100  method.lr=${lr[i]} method.num_epochs=${num_epochs[i]} +model=bart-base
done
# Population results
python run_IF_exp.py results_dir="/results/zsre" task='zsre' iterations=5 tr_bsz=4 te_bsz=32 n=${n[5]} n_test=200 device=cuda:2 data_seed=1 approx_method=${method[2]} model.pt="/models/zsre/model_zsre_1_4499_200" method.regularization_param=100 method.lr=${lr[2]} method.num_epochs=100 +model=bart-base

echo "Job completed at $(date)"




