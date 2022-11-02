# This file finetunes a base model on a subset of data.
# Can be used for BART base/zsRE or Distil GPT2/WikiText
import hydra
from pathlib import Path
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import os
import logging
import transformers
from src.utils import get_tokenizer, get_model
from src.ft_training import training
from src.data_zsre import extract_data_zsre
from src.data_wiki import extract_data_wiki


# Set directory
cwd = Path(__file__).parent
os.chdir(cwd)

# Set Config file
OmegaConf.register_new_resolver("uuid", lambda: utils.uuid())


@hydra.main(config_path='config', config_name='train_model')
def run(config):
    logging.info(f"Configuration: {config}")
    base_dir = hydra.utils.get_original_cwd()
    logging.info(f"Project base directory: {base_dir}")

    # Set seed for reproducibility
    random.seed(config.model_seed)
    np.random.seed(config.model_seed)
    torch.manual_seed(config.model_seed)

    # Download original model and tokenizer
    logging.info("Load Original Model")
    tokenizer = get_tokenizer(
        config.model.tokenizer_name, config.model.name, config.model.tokenizer_class)
    model_original = get_model(tokenizer, config.model.name, transformers, config.model.class_name, config.model.pt,
                            config.dropout, config.model.inner_params, config.no_grad_layers, config.half).to(config.device)
    if config.task == 'wiki':
        # Adding padding to tokenizer
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model_original.resize_token_embeddings(len(tokenizer))
        model_original.transformer.wte.weight.data[-1] = model_original.transformer.wte.weight.data.mean(
            0)

    # Download train/test data
    if config.task == "zsre":
        train_data, test_data = extract_data_zsre(
            config, model_original, tokenizer, base_dir)
    else:
        train_data, test_data = extract_data_wiki(
            config, tokenizer, base_dir)

    # Finetune model
    trained_model, training_logs = training(
        config, model_original, tokenizer, train_data, test_data, base_dir, lr=config.ft.lr)

    # Save Results
    results_dir = base_dir+config.results_dir
    results_path = f"{results_dir}/results_{config.task}_{config.data_seed}_{config.n}_{config.n_test}.pt"
    torch.save(training_logs, results_path)
    print("Results outputed to: ", results_path)


if __name__ == "__main__":
    run()
