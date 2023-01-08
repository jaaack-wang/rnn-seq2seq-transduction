'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang
Website: https://jaaack-wang.eu.org
About: Code for creating dataloader in PyTorch
'''
import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader

import sys
import pathlib
# import from local script
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils import read_data


class Transform(Dataset):
    '''Creates a Dataset class that encode data into sequences of integers.
    
    Args:
        - data (list): contains a list of [input, output].
        - in_seq_encoder (method): a function to encode input
        - out_seq_encoder (method): a function to encode output
    '''
    def __init__(self, data, in_seq_encoder, out_seq_encoder):
        self.data = data
        self.in_seq_encoder = in_seq_encoder
        self.out_seq_encoder = out_seq_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        in_seq, out_seq = self.data[index]
        x = self.in_seq_encoder(in_seq)
        y = self.out_seq_encoder(out_seq)
        
        return x, y


def make_map(vocab):
    '''Makes a vocab to idx and idx to vocab map.
    
    Args:
        - vocab (iterable): vocabulary to be used.
    '''
    v2idx= {"<s>": 0, "</s>": 1, "<unk>": 2}
    
    for v in vocab:
        v2idx.update({v: len(v2idx)})
    idx2v = {idx: v for v, idx in v2idx.items()}
    return v2idx, idx2v


def get_text_encoder_decoder(vocab, tokenizer=None):
    '''Returns two methods, a text encoder that converts
    text into a list of indices, and a text decoder that
    converts a list of indices into a list of tokens.
    
    Typically, in transduction tasks, there are no unknown tokens.
    For simplicity, the last index that equals the vocab size 
    is preserved for all unknown tokens. If a pad index is given,
    please add the token into vocab and specify the index when 
    using create_dataloader. Otherwise, the padding index also 
    defaults to the vocab size.
    
    Args:
        - vocab (iterable): vocabulary.  
        - tokenizer (method/None): a method that converts
          a text into a list of tokens. Defaults to None. 
    '''
    
    
    v2idx, idx2v = make_map(vocab)
    
    if not tokenizer:
        tokenizer = lambda t: t

    
    encoder = lambda text: [0] + [v2idx.get(t, v2idx["<unk>"])
                                  for t in tokenizer(text)] + [1]
    
    decoder = lambda idx: [idx2v.get(i, "<unk>") 
                           for i in idx]
    return encoder, decoder


def make_tensors(size, fill_idx):
    '''Makes tensors of a given size filled with a given fill_idx.
    The type of the returned tensors is torch.int64.'''
    tensors = torch.fill_(torch.empty(size), fill_idx)
    return tensors.long()


def collate_fn(batch, padding_idx, 
               in_max_seq_len=None, 
               out_max_seq_len=None):
    '''Collation function that collates a batch of data.
    
    Args: 
        - batch: transformed batched data.
        - padding_idx (integer): integers to pad a batch.
        - in/out_max_seq_len (None/integer): max in/output 
          sequence length. If the given length is shorter than
          the actual max in/output sequence length, the later is 
          used. Defaults to None, using the actual max length.
    '''
    
    N = len(batch)
    X, Y = zip(*batch)
    in_max_len = max([len(x) for x in X])
    out_max_len = max([len(y) for y in Y])
    
    if in_max_seq_len and in_max_seq_len > in_max_len:
        in_max_len = in_max_seq_len
    
    inputs = make_tensors((in_max_len, N), padding_idx)
    
    if out_max_seq_len and out_max_seq_len > out_max_len:
        out_max_len = out_max_seq_len
        
    outputs = make_tensors((out_max_len, N), padding_idx)

    for idx, (x, y) in enumerate(batch):
        inputs[:len(x), idx] = torch.Tensor(x).long()
        outputs[:len(y), idx] = torch.Tensor(y).long()

    return inputs, outputs


def create_dataloader(data,
                      in_seq_encoder,
                      out_seq_encoder, 
                      padding_idx, 
                      shuffle=False,
                      batch_size=256,
                      in_max_seq_len=None, 
                      out_max_seq_len=None):
    '''Creates a dataloader object. 
    
    Args:
        - data (list/str): contains a list of [input, output].
          Can also be a filepath path to the data. See read_data.
        - in_seq_encoder (method): a function to encode input.
        - out_seq_encoder (method): a function to encode output.
        - padding_idx (integer): integers to pad a batch.
        - shuffle (bool): whether to shuffle the data. Defaults to False.
        - batch_size(integer): batch size. Defaults to 256.
        - in/out_max_seq_len (None/integer): max in/output 
          sequence length. If the given length is shorter than
          the actual max in/output sequence length, the later is 
          used. Defaults to None, using the actual max length.
    '''

    if isinstance(data, str):
        data = read_data(data)
    
    collate = lambda batch: collate_fn(batch, padding_idx, 
                                       in_max_seq_len, 
                                       out_max_seq_len)
    dataset = Transform(data, in_seq_encoder, out_seq_encoder)
    dataloader = DataLoader(dataset, batch_size, 
                            shuffle, collate_fn=collate)
    return dataloader


def customize_dataloader_func(in_seq_encoder,
                              out_seq_encoder, 
                              padding_idx, 
                              shuffle=False,
                              batch_size=256,
                              in_max_seq_len=None, 
                              out_max_seq_len=None):
    
    return partial(create_dataloader, 
                   shuffle=shuffle, 
                   batch_size=batch_size, 
                   padding_idx=padding_idx, 
                   in_seq_encoder=in_seq_encoder, 
                   out_seq_encoder=out_seq_encoder, 
                   in_max_seq_len=in_max_seq_len, 
                   out_max_seq_len=out_max_seq_len)
