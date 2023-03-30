'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang
Website: https://jaaack-wang.eu.org
About: Utility functions for training, evaluation,
and deployment (i.e., prediction).
'''
import torch
import torch.nn as nn
import torch.nn.init as init
from functools import partial
import matplotlib.pyplot as plt

import sys
import pathlib
# import from local script
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from model import *


def init_weights(model, init_method=init.xavier_uniform_):
    '''Initialize model's weights by a given method. Defaults to
        Xavier initialization.'''
    
    for name, param in model.named_parameters():
        if 'weight' in name:
            init_method(param.data)
        else:
            init.constant_(param.data, 0)


def count_parameters(model):
    '''Count the number of trainable parameters.'''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(ModelConfig, init_method=init.xavier_uniform_):
    '''Customized function to initialze a model given ModelConfig.'''
    rnn_type = ModelConfig['rnn_type']
    hidden_size = ModelConfig['hidden_size']
    embd_dim = ModelConfig['embd_dim']
    num_layers = ModelConfig['num_layers']
    dropout_rate = ModelConfig['dropout_rate']
    use_attention = ModelConfig['use_attention']
    bidirectional = ModelConfig['bidirectional']
    in_vocab_size = ModelConfig['in_vocab_size']
    out_vocab_size = ModelConfig['out_vocab_size']
    
    device = torch.device(ModelConfig["device"])
    
    reduction_method = ModelConfig['reduction_method'] 
    if reduction_method == "sum":
        reduction_method = torch.sum
    elif reduction_method == "mean":
        reduction_method = torch.mean
    else:
        raise TypeError(f"unknown reduction method: {reduction_method}")

    encoder = Encoder(in_vocab_size, hidden_size, 
                      embd_dim, num_layers, rnn_type,
                      dropout_rate, bidirectional, 
                      reduction_method)

    attention = Attention(hidden_size)
    decoder = Decoder(out_vocab_size, hidden_size, 
                      embd_dim, num_layers, rnn_type,
                      attention, use_attention, 
                      dropout_rate)

    model = Seq2Seq(encoder, decoder, device).to(device)
    
    if init_method != None:
        init_weights(model, init_method)
    
    n = count_parameters(model)
    print(f'The model has {n:,} trainable parameters')
    return model


def metrics(Y, Ypred):
    '''Computer the following three metrics:
        - full sequence accuracy: % of sequences correctly generated from end to end
        - first n-symbol accuracy: % of first n symbols correctly generated
        - overlap rate: % of pairwise overlapping symbols
    '''
    # pairwise overlap
    pairwise_overlap = (Y == Ypred).to(torch.float64)
    # pairwise overlap over across sequences (within the given batch)
    per_seq_overlap = pairwise_overlap.mean(dim=0)
    # overlap rate
    overlap_rate = per_seq_overlap.mean().item()
    # full sequence accuracy
    abs_correct = per_seq_overlap.isclose(torch.tensor(1.0, dtype=torch.float64))
    full_seq_accu = abs_correct.to(torch.float64).mean().item()

    # if the n-th symbol does not match, set the following overlapping values to 0
    if pairwise_overlap.dim() <= 1:
        min_idx = pairwise_overlap.argmin(0)
        if pairwise_overlap[min_idx] == 0:
            pairwise_overlap[min_idx:] = 0
        
    else:
        for col_idx, min_idx in enumerate(pairwise_overlap.argmin(0)):
            if pairwise_overlap[min_idx, col_idx] == 0:
                pairwise_overlap[min_idx:, col_idx] = 0
                
    # first n-symbol accuracy 
    first_n_accu = pairwise_overlap.mean().item()
    
    return full_seq_accu, first_n_accu, overlap_rate


def _get_results(dic):
    loss = dic["loss"]
    overlap_rate = dic["overlap rate"]
    full_seq_acc = dic["full sequence accuracy"]
    first_n_acc = dic["first n-symbol accuracy"]
    return [loss, full_seq_acc, first_n_acc, overlap_rate]


def get_results(log, train_log=True):
    '''Return the results of the four metrics: loss, 
    full sequence accuracy, first n-symbol accuracy, 
    overlap rate, given a result dictionary.'''
    
    if train_log:
        best = log["Best eval accu"]
        best_train = best["Train"]
        best_dev = best["Eval"]
        train_res = _get_results(best_train)
        dev_res = _get_results(best_dev)
        return train_res, dev_res
    
    return _get_results(log)


def evaluate(model, dataloader, criterion, 
             per_seq_len_performance=False):
    '''Evaluate model performance on a given dataloader.
    "per_seq_len_performance" can be reported if each batch
    in the dataloader only consists of a specific length.
    '''
    model.eval()
    
    if per_seq_len_performance:
        seq_len = set(X.shape[0] for X, _ in dataloader)
        assert len(seq_len) == len(dataloader), "Each batch" \
        " must contain sequences of a specific length. "
        
        perf_log = dict()
        
    # aggragate performance
    aggr_perf = {"loss": 0.0, 
                 "full sequence accuracy": 0.0, 
                 "first n-symbol accuracy": 0.0, 
                 "overlap rate": 0.0}
    
    with torch.no_grad():
        for X, Y in dataloader:
            x_seq_len = X.shape[0] - 2 # not counting <s> and </s>
            seq_len, batch_size = Y.shape
            seq_len -= 1 # logits does not have <s>

            X = X.to(model.device)
            Y = Y.to(model.device)
            logits, _ = model(X, Y, teacher_forcing_ratio=0.0)
            
            Ypred = logits.view(seq_len, batch_size, -1).argmax(2)
            full_seq_accu, first_n_accu, overlap_rate = metrics(Y[1:], Ypred)
            loss = criterion(logits, Y[1:].view(-1))
            
            aggr_perf["loss"] += loss.item()
            aggr_perf["full sequence accuracy"] += full_seq_accu
            aggr_perf["first n-symbol accuracy"] += first_n_accu
            aggr_perf["overlap rate"] += overlap_rate
            
            if per_seq_len_performance:
                perf_log[f"Len-{x_seq_len}"] = {"loss": loss.item(), 
                                                "full sequence accuracy": full_seq_accu, 
                                                "first n-symbol accuracy": first_n_accu, 
                                                "overlap rate": overlap_rate}
            
    aggr_perf = {k:v/len(dataloader) for k,v in aggr_perf.items()}
    
    if per_seq_len_performance:
        perf_log[f"Aggregated"] = aggr_perf
        
        return aggr_perf, perf_log
            
    return aggr_perf

            

def train_loop(model, dataloader, optimizer, criterion, teacher_forcing_ratio):
    '''A single training loop (for am epoch). 
    '''
    model.train()
    
    for X, Y in dataloader:
        seq_len, batch_size = Y.shape
        seq_len -= 1 # logits does not have <s>
        
        X = X.to(model.device)
        Y = Y.to(model.device)
        optimizer.zero_grad()
        
        logits, _ = model(X, Y, teacher_forcing_ratio)
        Ypred = logits.view(seq_len, batch_size, -1).argmax(2)        
        loss = criterion(logits, Y[1:].view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()


def train_and_evaluate(model, train_dl, eval_dl, 
                       criterion, optimizer, 
                       saved_model_fp="model.pt", 
                       acc_threshold=0.0, 
                       print_eval_freq=5, 
                       max_epoch_num=10, 
                       train_exit_acc=1.0, 
                       eval_exit_acc=1.0,
                       teacher_forcing_ratio=1.0):
    '''Trains and evaluates model while training and returns 
    the training log. The best model with highest full sequence 
    accuracy is saved and returned.
    
    Args:
        - model (nn.Module): a neural network model in PyTorch.
        - train_dl (Dataset): train set dataloader.
        - eval_dl (Dataset): dataloader for evaluation.
        - criterion (method): loss function for computing loss.
        - optimizer (method): Optimization method.
        - saved_model_fp (str): filepath for the saved model (.pt).
        - acc_threshold (float): the min accuracy to save model. 
          Defaults to 0.0. If set greater than 1, no model will be saved.
        - print_eval_freq (int): print and evaluation frequency.
        - max_epoch_num (int): max epoch number. Defaults to 10. Training 
          is stopped if the max epoch number is run out. 
        - train_exit_acc (float): the min train accuracy to exit training.
          Defaults to 1.0. Only takes effect if eval_exit_acc is also met.
        - eval_exit_acc (float): the min eval accu to exit training. Defaults 
          to 1.0. Training is stopped if the eval accuracy if 1.0 or both
          train_exit_acc and eval_exit_acc are met.
        - teacher_forcing_ratio (float): the probability of using the real next
          symbols from the output sequences at decoding time during training.
    '''
    
    log = dict()
    best_acc, best_epoch = acc_threshold, 0
    epoch, train_acc, eval_acc = 0, 0, 0
    
    while (epoch < max_epoch_num) and (eval_acc != 1.0) and (
        train_acc < train_exit_acc or eval_acc < eval_exit_acc):
        
        epoch += 1
        
        train_loop(model, train_dl, optimizer, criterion,
                   teacher_forcing_ratio)
        
        if epoch % print_eval_freq == 0:
            
            train_perf = evaluate(model, train_dl, criterion)
            train_acc = train_perf['full sequence accuracy']
            
            eval_perf = evaluate(model, eval_dl, criterion)
            eval_acc = eval_perf['full sequence accuracy']
            
            print(f"Current epoch: {epoch}, \ntraining performance: " \
                  f"{train_perf}\nevaluation performance: {eval_perf}\n")
            
            log[f"Epoch#{epoch}"] = {"Train": train_perf, "Eval": eval_perf}
        
            if eval_acc > best_acc:
                best_acc = eval_acc
                best_epoch = epoch
                torch.save(model.state_dict(), saved_model_fp)
    
    if best_acc > acc_threshold:
        log["Best eval accu"] = {"Epoch Number": epoch}
        log["Best eval accu"].update(log[f"Epoch#{best_epoch}"])
        print(saved_model_fp + " saved!\n")
        model.load_state_dict(torch.load(saved_model_fp))
        
    return log


def predict(text, model, in_seq_encoder, out_seq_decoder, 
            in_seq_decoder=None, max_output_len=None, 
            visualize=True, show_plot=True, saved_plot_fp=None):
    
    if isinstance(text, str):
        pass
    elif isinstance(text, (list, tuple,)):
        assert all(isinstance(t, str) 
                   for t in text), "must be a list of strs"
        
        output_seqs, attn_weights = [], []
        
        for t in text:
            o, w = predict(t, model, in_seq_encoder, 
                           out_seq_decoder, visualize=False, 
                           max_output_len=max_output_len)
            output_seqs.append(o)
            attn_weights.append(w)

        return output_seqs, attn_weights

    else:
        raise TypeError("texts must be a str or a list of strs," \
                        f" {type(text)} was given.")
    
    device = model.device
    in_seq = in_seq_encoder(text)
    in_seq_tensor = torch.Tensor(in_seq).long().unsqueeze(1).to(device)

    model.eval()
    y = torch.Tensor([[0]]).long().to(device)
    outputs, attn_ws = [], []
    encoder_outputs, hidden, cell  = model.encoder(in_seq_tensor)
    
    if max_output_len == None:
        max_output_len = len(in_seq) + 3
    
    while y.item() != 1 and len(outputs) < max_output_len:
        output, hidden, cell, attn_w = \
                model.decoder(y, hidden, cell, encoder_outputs)
        y = output.argmax(1).unsqueeze(0)
        outputs.append(y.item()); attn_ws.append(attn_w)
    
    if attn_ws[0] != None:
        attn_ws = torch.cat(attn_ws).squeeze(1)

        if device.type != "cpu":
            attn_ws = attn_ws.cpu().detach().numpy()
        else:
            attn_ws = attn_ws.detach().numpy()
    else:
        visualize = False
    
    output_seq = out_seq_decoder(outputs)
    
    if visualize:
        if in_seq_decoder == None:
            in_seq_decoder = out_seq_decoder
        
        in_seq_len, out_seq_len = len(in_seq)-1, len(outputs)
        
        width = max(int(in_seq_len * 0.3), 1)
        height = max(int(out_seq_len * 0.3), 1)
        plt.figure(figsize=(width, height))
        plt.imshow(attn_ws[:, 1:], cmap='BuGn')
        plt.xticks(range(in_seq_len), 
                   in_seq_decoder(in_seq)[1:], rotation=45)
        plt.yticks(range(out_seq_len), output_seq)
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.grid(True, alpha=0.05)
        
        if saved_plot_fp != None:
            plt.savefig(saved_plot_fp, dpi=600, bbox_inches='tight')
            
        if show_plot:
            plt.show()
        else:
            plt.close()

    return output_seq, attn_ws


def customize_predictor(model, in_seq_encoder, out_seq_decoder, 
                        in_seq_decoder=None, max_output_len=None, 
                        visualize=True, show_plot=True, saved_plot_fp=None):
    '''Customize a predictor function so that the func can be used more easily.'''
    return partial(predict, model=model,
                   in_seq_encoder=in_seq_encoder, 
                   out_seq_decoder=out_seq_decoder,
                   in_seq_decoder=in_seq_decoder,
                   max_output_len=max_output_len,
                   visualize=visualize, 
                   show_plot=show_plot, 
                   saved_plot_fp=saved_plot_fp)
