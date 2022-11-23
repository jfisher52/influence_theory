# Finetune Distill GPT2 using subset of WikiText
n=(40 105 275 724 1903)

for i in 0 1 2 3 4
do
    python run_train_model.py results_dir="/models/wiki" task='wiki' batch_size=4 iterations=50 n=${n[i]} n_test=200 device=cuda:3 data_seed=1 ft.lr=1e-5 +model=gpt2 
done

echo "Job completed at $(date)"