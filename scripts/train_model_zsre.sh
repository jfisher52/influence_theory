# Finetune Bart base using subset of zsRE
for n in 49 122 182 302 743 4499
do
    for n_test in 200
    do
        python run_train_model.py results_dir="/models/zsre" task='zsre' batch_size=16 iterations=50 n=${n} n_test=${n_test} device=cuda:2 data_seed=1 ft.lr=1e-6 +model=bart-base 
    done
done
