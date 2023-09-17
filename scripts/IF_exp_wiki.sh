# Run all WikiText approxmiation of influence function experiements
# Set parameters
lr=(1e-2 1e-2 1e-3 1e-3)
num_epochs=(100 50 25 50)
method=(conjugate_gradient sgd svrg arnoldi identity)
n=(40 105 275 724 1903)

# Run experiements
for i in 0 1 2 3
do
    python run_IF_exp.py results_dir="/results/wiki" task='wiki' iterations=5 tr_bsz=1 te_bsz=2 n=${n[0]} n_test=200 device=cuda:3 data_seed=1 approx_method=${method[i]} model.pt="/models/wiki/model_wiki_1_40_200" method.regularization_param=1 method.num_epochs=${num_epochs[i]} method.lr=${lr[i]} +model=gpt2 
    python run_IF_exp.py results_dir="/results/wiki" task='wiki' iterations=5 tr_bsz=1 te_bsz=2 n=${n[1]} n_test=200 device=cuda:3 data_seed=1 approx_method=${method[i]} model.pt="/models/wiki/model_wiki_1_105_200" method.regularization_param=1 method.num_epochs=${num_epochs[i]} method.lr=${lr[i]} +model=gpt2 
    python run_IF_exp.py results_dir="/results/wiki" task='wiki' iterations=5 tr_bsz=1 te_bsz=2 n=${n[2]} n_test=200 device=cuda:3 data_seed=1 approx_method=${method[i]} model.pt="/models/wiki/model_wiki_1_275_200" method.regularization_param=1 method.num_epochs=${num_epochs[i]} method.lr=${lr[i]} +model=gpt2 
    python run_IF_exp.py results_dir="/results/wiki" task='wiki' iterations=5 tr_bsz=1 te_bsz=2 n=${n[3]} n_test=200 device=cuda:3 data_seed=1 approx_method=${method[i]} model.pt="/models/wiki/model_wiki_1_724_200" method.regularization_param=1 method.num_epochs=${num_epochs[i]} method.lr=${lr[i]} +model=gpt2 
done
# Population results
python run_IF_exp.py results_dir="/results/wiki" task='wiki' iterations=5 tr_bsz=1 te_bsz=2 n=${n[4]} n_test=200 device=cuda:3 data_seed=1 approx_method=${method[2]} model.pt="/models/wiki/model_wiki_1_1903_200" method.regularization_param=1 method.num_epochs=100 method.lr=${lr[2]} +model=gpt2

echo "Job completed at $(date)"



