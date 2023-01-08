'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang
Website: https://jaaack-wang.eu.org
About: RNN Seq2Seq models (Simple RNN, GRU, LSTM)
in PyTorch. Allows: attention, bidirectional RNN,
as well as multilayered RNN etc.
'''
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, 
                 in_vocab_size, hidden_size, 
                 embd_dim, num_layers=1, rnn_type="SRNN",
                 dropout_rate=0.0, bidirectional=False, 
                 reduction_method=torch.sum):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(in_vocab_size, embd_dim)
        
        self.num_layers = num_layers
        self.rnn_type = rnn_type.upper()
        self.bidirectional = bidirectional
        if self.rnn_type == "GRU": rnn_ = nn.GRU
        elif self.rnn_type == "LSTM": rnn_ = nn.LSTM
        elif self.rnn_type == "SRNN": rnn_ = nn.RNN
        else: raise ValueError("Only supports SRNN, GRU, LSTM," \
                               " but {self.rnn_type} was given.")
        self.rnn = rnn_(embd_dim, 
                        hidden_size, 
                        num_layers,
                        bidirectional=bidirectional)
        self.reduce = reduction_method
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, X):      
        # X: (max input seq len, batch size)
        # embd: (max input seq len, batch size, embd dim)
        embd = self.dropout(self.embedding(X))   
        
        # outputs: (max input seq len, batch size, 
        #                     hidden size * num directions)
        # hidden: (num directions * num layers, batch size, hidden size)
        # cell: (num directions * num layers, batch size, hidden size)
        if self.rnn_type == "LSTM":
            outputs, (hidden, cell) = self.rnn(embd)
        else:
            outputs, hidden = self.rnn(embd)
            cell = None # placeholder
         
        if self.bidirectional:
            seq_len, batch_size = X.shape
            
            hidden = hidden.view(2, self.num_layers, batch_size, -1)
            hidden = self.reduce(hidden, dim=0)
            
            if self.rnn_type == "LSTM":
                cell = cell.view(2, self.num_layers, batch_size, -1)
                cell = self.reduce(cell, dim=0)
            
            outputs = outputs.view(seq_len, batch_size, 2, -1)
            outputs = self.reduce(outputs, dim=2)
        
        # outputs: (max input seq len, batch size, hidden size)
        # hidden: (num layers, batch size, hidden size)
        # cell: (num layers, batch size, hidden size)
        return outputs, hidden, cell


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.attn = nn.Linear(2 * hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs):
        # hidden: (batch size, hidden size)
        # encoder_outputs: (max input seq len, 
        #                          batch size, hidden size)
        seq_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]
        
        # hidden: (batch size, max input seq len, hidden size)
        # encoder_outputs: same as hidden above 
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # cat: (batch size, max input seq len, 2 * hidden size)
        # energy: (batch size, max input seq len, hidden size)
        # attention: (batch size, max input seq len)
        cat = torch.cat((hidden, encoder_outputs), dim = 2) 
        energy = torch.tanh(self.attn(cat))
        attention = self.v(energy).squeeze(2)
        
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, 
                 out_vocab_size, hidden_size, 
                 embd_dim, num_layers=1, rnn_type="RNN", 
                 attention=None, use_attention=True, 
                 dropout_rate=0.0, reduction_method=torch.sum):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(out_vocab_size, embd_dim)
        
        self.rnn_type = rnn_type.upper()
        self.use_attention = use_attention
        if self.rnn_type == "GRU": rnn_ = nn.GRU
        elif self.rnn_type == "LSTM": rnn_ = nn.LSTM
        elif self.rnn_type == "SRNN": rnn_ = nn.RNN
        else: raise ValueError("Only supports SRNN, GRU, LSTM," \
                               " but {self.rnn_type} was given.")
        if use_attention:
            self.rnn = rnn_(embd_dim + hidden_size, 
                            hidden_size, num_layers)
        else:
            self.rnn = rnn_(embd_dim, hidden_size, num_layers)
        
        self.attention = attention
        self.reduce = reduction_method
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(hidden_size, out_vocab_size)
        
    def forward(self, y, hidden, cell, encoder_outputs):
        # y: (1, batch size)
        # hidden: (num layers, batch size, hidden size)
        # cell: (num layers, batch size, hidden size) or a 3-D placeholder 
        # encoder_outputs: (max input seq len, batch size, hidden size)
        
        # embd: (num layers, batch size, embd dim)
        embd = self.dropout(self.embedding(y))
        
        if self.use_attention and self.attention:
            # reduced_hidden: (batch size, hidden size)
            # attn_weights: (batch size, 1, max input seq len)
            reduced_hidden = self.reduce(hidden, dim=0)
            attn_weights = self.attention(reduced_hidden, 
                                          encoder_outputs).unsqueeze(1)

            # encoder_outputs: (batch size, max input seq len, hidden size)
            encoder_outputs = encoder_outputs.permute(1, 0, 2)

            # weighted: (1, batch size, hidden size)
            weighted = torch.bmm(attn_weights, encoder_outputs).permute(1, 0, 2)

            # cat: (1, batch size, embd dim + hidden size)
            rnn_input = torch.cat((embd, weighted), dim = 2) 
        else:
            rnn_input = embd
            attn_weights = None # placeholder
        
        # hidden/cell: (num layers, batch size, hidden size)
        # output: (1, batch size, hidden size)
        if self.rnn_type == "LSTM":
            output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        else:
            output, hidden = self.rnn(rnn_input, hidden)
                
        # output: (batch size, out vocab size)
        output = self.fc_out(output.squeeze(0))        
        return output, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, X, Y, teacher_forcing_ratio=0.0):
        # X: (max input seq len, batch size)
        # Y: (max output seq len, batch size)

        y = Y[0:1] # y: (1, batch size)
        outputs, attn_weights = [], []
        encoder_outputs, hidden, cell = self.encoder(X)
        for t in range(1, Y.shape[0]):
            output, hidden, cell, attn_w = \
            self.decoder(y, hidden, cell, encoder_outputs)
            
            outputs.append(output); attn_weights.append(attn_w)
            
            teacher_force = random.random() < teacher_forcing_ratio
            if teacher_force: y = Y[t:t+1] 
            else: y = output.argmax(1).unsqueeze(0)
        
        # outputs: ((max output seq len-1) * batch size, out vocab size)
        outputs = torch.cat(outputs).to(self.device)
        return outputs, attn_weights
