# Script to finetune language models for experiements
# Finetune Bart base using subset of zsRE
for n in 49 122 302 743 182 4499
do
    for n_test in 200
    do
        python run_train_model.py results_dir="/models/zsre" task='zsre' iterations=20 n=${n} n_test=${n_test} device=cuda:1 data_seed=1 ft.lr=1e-6 +model=bart-base 
    done
done

# Finetune Distill GPT2 using subset of WikiText
for n in 40 105 275 724 1903
do
    for n_test in 200
    do
        python run_train_model.py results_dir="/models/wiki" task='wiki' batch_size=8 iterations=20 n=${n} n_test=${n_test} device=cuda:1 data_seed=1 ft.lr=1e-6 +model=gpt2
    done
done

echo "Job completed at $(date)"