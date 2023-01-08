'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang
Website: https://jaaack-wang.eu.org
About: Customized training pipeline.
'''
import random
import argparse
from os.path import join

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

import sys
import pathlib
# import from local script
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from visualization import plot_training_log
from utils import read_data, create_dir, read_json, save_dict_as_json
from dataloader import get_text_encoder_decoder, customize_dataloader_func
from pytorch_utils import get_model, train_and_evaluate, count_parameters


parser = argparse.ArgumentParser()
parser.add_argument("--config_fp", default='config.json', type=str, help="The filepath to the config.json.")
args = parser.parse_args()
config = read_json(args.config_fp)


def main():
    ModelConfig = config["ModelConfig"]
    TrainConfig = config["TrainConfig"]
    
    train = read_data(TrainConfig["train_fp"])
    if TrainConfig["dev_fp"]:
        dev = read_data(TrainConfig["dev_fp"])
    else:
        random.shuffle(train)
        split = int(len(train) * TrainConfig["split"])
        train, dev = train[:split], train[split:]
    
    if TrainConfig["vocab_fp"]:
        vocab = read_json(TrainConfig["vocab_fp"])
        in_vocab = vocab["in_vocab"]
        out_vocab = vocab["out_vocab"]
    else:
        vocab = None
    
    if TrainConfig["x_sep"]:
        train = [[t1.split(TrainConfig["x_sep"]), t2] for (t1, t2) in train]
        dev = [[t1.split(TrainConfig["x_sep"]), t2] for (t1, t2) in dev]
    
    if TrainConfig["y_sep"]:
        train = [[t1, t2.split(TrainConfig["y_sep"])] for (t1, t2) in train]
        dev = [[t1, t2.split(TrainConfig["y_sep"])] for (t1, t2) in dev]
    
    if not vocab:
        in_vocab = list(set([v for (text, _) in train for v in text]))
        out_vocab = list(set([v for (_, text) in train for v in text]))
    
    padding_idx = 1
    TrainConfig["in_vocab"] = in_vocab
    TrainConfig["out_vocab"] = out_vocab
    ModelConfig["in_vocab_size"] = len(in_vocab) + 3
    ModelConfig["out_vocab_size"] = len(out_vocab) + 3
    batch_size = TrainConfig["batch_size"]
        
    in_seq_encoder, _ = get_text_encoder_decoder(in_vocab)
    out_seq_encoder, _ = get_text_encoder_decoder(out_vocab)
    dataloader_func = customize_dataloader_func(in_seq_encoder, out_seq_encoder, 
                                                padding_idx, batch_size=batch_size)
    
    train_dl = dataloader_func(train)
    dev_dl = dataloader_func(dev)
    
    device = ModelConfig["device"]
    if device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ModelConfig["device"] = device
    
    model = get_model(ModelConfig)
    n_param = count_parameters(model)
    ModelConfig["param#"] = n_param
    
    lr = TrainConfig["learning_rate"]
    weight_decay = TrainConfig["weight_decay"]
    acc_threshold = TrainConfig["acc_threshold"]
    print_eval_freq = TrainConfig["print_eval_freq"]
    max_epoch_num = TrainConfig["max_epoch_num"]
    train_exit_acc = TrainConfig["train_exit_acc"]
    eval_exit_acc = TrainConfig["eval_exit_acc"]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=lr, weight_decay=weight_decay)
    
    out_folder = TrainConfig['output_folder_name']
    sub_folder = TrainConfig["sub_folder_name"]
    if sub_folder == None:
        sub_folder = datetime.now().strftime('%Y-%m-%d~%H-%M-%S')
    
    out_folder = join(out_folder, sub_folder)
    create_dir(out_folder)
    saved_model_fp = join(out_folder, "model.pt")
    log = train_and_evaluate(model, train_dl, dev_dl, 
                             criterion, optimizer, 
                             saved_model_fp=saved_model_fp, 
                             acc_threshold=acc_threshold, 
                             print_eval_freq=print_eval_freq, 
                             max_epoch_num=max_epoch_num, 
                             train_exit_acc=train_exit_acc, 
                             eval_exit_acc=eval_exit_acc)
    
    ModelConfig['device'] = device
    save_dict_as_json(log, join(out_folder, "training_log.json"))
    save_dict_as_json(ModelConfig, join(out_folder, "ModelConfig.json"))
    save_dict_as_json(TrainConfig, join(out_folder, "TrainConfig.json"))
    
    if TrainConfig["plot_training_log"]:
        show_plot = TrainConfig["show_plot"]
        saved_plot_fp = join(out_folder, "training_log.png")
        plot_training_log(log, show_plot, saved_plot_fp)


if __name__ == "__main__":
    main()
