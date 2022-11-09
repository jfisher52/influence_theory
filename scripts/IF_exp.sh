# Run all approxmiation of influence function experiements
# zsRE experiements
for method in conjugate_gradient sgd svrg arnoldi
do
    python run_IF_exp.py results_dir="run_approx_IF" task='zsre' iterations=5 n=49 n_test=200 device=cuda:2 data_seed=1 approx_method=${method} model.pt="/models/zsre/model_zsre_1_49_200" method.regularization_param=100  method.sgd.lr=5e-4  method.svrg.lr=5e-4  method.arnoldi.n_it=30 +model=bart-base
    python run_IF_exp.py results_dir="run_approx_IF" task='zsre' iterations=5 n=122 n_test=200 device=cuda:2 data_seed=1 approx_method=${method} model.pt="/models/zsre/model_zsre_1_122_200" method.regularization_param=100  method.sgd.lr=5e-4 method.svrg.lr=5e-4  method.arnoldi.n_it=30 +model=bart-base 
    python run_IF_exp.py results_dir="run_approx_IF" task='zsre' iterations=5 n=182 n_test=200 device=cuda:2 data_seed=1 approx_method=${method} model.pt="/models/zsre/model_zsre_1_182_200" method.regularization_param=100  method.sgd.lr=5e-4  method.svrg.lr=5e-4  method.arnoldi.n_it=30 +model=bart-base
    python run_IF_exp.py results_dir="run_approx_IF" task='zsre' iterations=5 n=302 n_test=200 device=cuda:2 data_seed=1 approx_method=${method} model.pt="/models/zsre/model_zsre_1_302_200" method.regularization_param=100  method.sgd.lr=5e-4  method.svrg.lr=5e-4  method.arnoldi.n_it=30 +model=bart-base
    python run_IF_exp.py results_dir="run_approx_IF" task='zsre' iterations=5 n=743 n_test=200 device=cuda:2 data_seed=1 approx_method=${method} model.pt="/models/zsre/model_zsre_1_743_200" method.regularization_param=100  method.sgd.lr=5e-4  method.svrg.lr=5e-4  method.arnoldi.n_it=30 +model=bart-base
done
echo "Job completed at $(date)"
python run_IF_exp.py results_dir="run_approx_IF" task='zsre' iterations=5 n=4499 n_test=200 device=cuda:2 data_seed=1 approx_method='svrg' model.pt="/models/zsre/model_zsre_1_4499_200" method.regularization_param=100 method.svrg.lr=5e-4 method.svrg.num_epochs=100 +model=bart-base

# WikiText experiements
for method in conjugate_gradient sgd svrg arnoldi
do
    python run_IF_exp.py results_dir="run_approx_IF" task='wiki' iterations=5 n=40 n_test=200 device=cuda:2 data_seed=1 approx_method=${method} model.pt="/models/wiki/model_wiki_1_40_200"  method.regularization_param=1 method.sgd.lr=1e-2 method.svrg.lr=5e-3 method.arnoldi.n_it=50 +model=gpt2
    python run_IF_exp.py results_dir="run_approx_IF" task='wiki' iterations=5 n=105 n_test=200 device=cuda:2 data_seed=1 approx_method=${method} model.pt="/models/wiki/model_wiki_1_105_200"  method.regularization_param=1 method.sgd.lr=1e-2 method.svrg.lr=5e-3 method.arnoldi.n_it=50 +model=gpt2
    python run_IF_exp.py results_dir="run_approx_IF" task='wiki' iterations=5 n=275 n_test=200 device=cuda:2 data_seed=1 approx_method=${method} model.pt="/models/wiki/model_wiki_1_275_200"  method.regularization_param=1 method.sgd.lr=1e-2 method.svrg.lr=5e-3 method.arnoldi.n_it=50 +model=gpt2
    python run_IF_exp.py results_dir="run_approx_IF" task='wiki' iterations=5 n=724 n_test=200 device=cuda:2 data_seed=1 approx_method=${method} model.pt="/models/wiki/model_wiki_1_724_200" method.regularization_param=1 method.sgd.lr=1e-2 method.svrg.lr=5e-3 method.arnoldi.n_it=50 +model=gpt2
done
echo "Job completed at $(date)"
python run_IF_exp.py results_dir="run_approx_IF" task='wiki' iterations=5 n=1903 n_test=200 device=cuda:2 data_seed=1 approx_method='svrg' model.pt="/models/wiki/model_wiki_1_1903_200" method.regularization_param=1 method.svrg.lr=5e-3 method.svrg.num_epochs=100 +model=gpt2

