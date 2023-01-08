## Description

This repository presents a general framework for using RNN seq2seq models in modelling language transduction tasks. Customized pipelines for training and deploying your own models given properly formatted data and a corresponding `config.json` file are also provided. Tutorials for building your own pipelines or understanding the code may come later. 

This repository is a result of my project [rnn-seq2seq-learning](https://github.com/jaaack-wang/rnn-seq2seq-learning). Also see [rnn-transduction](https://github.com/jaaack-wang/rnn-transduction).



## Usage

**Requirements**: PyTorch (I used version 1.10.1), Matplotlib (I used version 3.4.2)

To use the customized pipelines for training and deploying your own models, you need to have training data that consists of input-output sequence pairs separated by a tab (i.e., `\t`). Moreover, both the input and output sequences should be presented in a way such that either each symbol in the sequence is a token already or the tokens can be easily inferred from a separator that is not part of the token list. If the latter is the case, you need to specify the separator(s) used in your `config.json` file, which is responsible for configuring the training and inference processes.

The `config.json` file can be created by using the templates provided in `example_configs`. The `config.json` file contains three types of configurations, one for the RNN model (i.e., `ModelConfig`), one for the training (i.e., `TrainConfig`), and one for the inference (i.e., `InferConfig`). If you use a separator in your data file, this information should be incorporated in the `TrainConfig` for training and `InferConfig` for depolyment. There are three types of RNN models available, i.e., SRNN (Simple RNN), GRU (Gated Recurrent Unit), and LSTM (Long Short-Term Memory).

To run the training pipeline, use the following command line. The outputs of the training are: a best model saved during training, training logs (including an optional plot), and the used configurations that will be re-used for inference. 

```cmd
python3 scripts/train.py --config_fp filepath_to_the_config_file
```

To run the inference pipeline, make sure you have a text file that consists of only the input sequences that you want to transduce (one example per line) and use the following command line:

```cmd
python3 scripts/predict.py --config_fp filepath_to_the_config_file
```

For example, you can do the following to test both pipelines on the toy examples provided by this repo:

```cmd
python3 scripts/train.py --config_fp example_configs/identity_config.json
python3 scripts/predict.py --config_fp example_configs/identity_config.json
```

Both pipelines are highly customized as a tradeoff for simplicity. 



## Notes

RNN seq2seq models are more flexible in modelling input-output transduction from end to end without needing aligned parallel examples, because the encoder and decoder are both RNNs that can process and produce variable-length sequences, respectively. However, for RNN seq2seq models, the problem lies in how to utilize the information encoded by the encoder during the decoding phase. For example, learning identity function is a hard problem for a pure RNN seq2seq model for at least two reasons. First, the information passed to the decoder contains vanishing information about the initial part of the input sequence, so it becomes impossible for the decoder to recall the initial input symbols as the input sequence becomes longer. Second, unlike RNNs, which produce an output symbol given an input symbol **by design**, the decoder of a pure RNN seq2seq model needs to figure out when to stop **by learning**. In this sense, pure RNN seq2seq models are not necessarily more powerful than RNN models.

In real world, however, nobody uses pure RNN seq2seq models to model anything. Instead, attention is used, which allows the decoder to access past information directly from the encoder. However, RNN seq2seq models with such a mechanism are more data-hungry and yet still have problem reliably generalizing to things that are dissimilar to the training data. For instance, their ability to capture identity function is arguably no better than the RNN models.

**The point here is that RNN models and RNN seq2seq models are different in their architectural designs, which come with different inductive biases, so they are suitable for learning different classes of transduction tasks**. It is not wise to use RNN seq2seq models to learn things we know RNN models are better at, although the former are much more convenient and flexible to use and usually yield better empirical results (with attention). Likewise, when RNN models are inpractical to use, go for other options.