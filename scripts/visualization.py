'''
Author: Zhengxiang (Jack) Wang 
GitHub: https://github.com/jaaack-wang
Website: https://jaaack-wang.eu.org
About: visualization function.
'''
import matplotlib.pyplot as plt


def plot_training_log(log, show_plot=True, saved_plot_fp=None):
    log = log.copy()
    if "Best eval accu" in log:
        log.pop("Best eval accu")
    
    train_loss, dev_loss = [], []
    train_full_seq_acc, dev_full_seq_acc = [], []
    train_first_n_acc, dev_first_n_acc = [], []
    train_overlap_rate, dev_overlap_rate = [], []
    epoch_nums = [int(e_n.split("#")[-1]) for e_n in log.keys()]

    for epoch in epoch_nums:
        train = log[f"Epoch#{epoch}"]["Train"]
        dev = log[f"Epoch#{epoch}"]["Eval"]
        train_loss.append(train['loss'])
        train_full_seq_acc.append(train["full sequence accuracy"])
        train_first_n_acc.append(train["first n-symbol accuracy"])
        train_overlap_rate.append(train["overlap rate"])

        dev_loss.append(dev['loss'])
        dev_full_seq_acc.append(dev["full sequence accuracy"])
        dev_first_n_acc.append(dev["first n-symbol accuracy"])
        dev_overlap_rate.append(dev["overlap rate"])

    
    fig = plt.figure(figsize=(16, 8))
    fig.subplots_adjust(hspace=0.3, wspace=0.2)
    
    ax1 = plt.subplot(221)
    ax1.plot(epoch_nums, train_loss, label="train")
    ax1.plot(epoch_nums, dev_loss, label="dev")
    ax1.set_xlabel("Epoch Number")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.1)
    ax1.legend()

    ax2 = plt.subplot(222)
    ax2.plot(epoch_nums, train_full_seq_acc, label="train" )
    ax2.plot(epoch_nums, dev_full_seq_acc, label="dev")
    ax2.set_xlabel("Epoch Number")
    ax2.set_ylabel("full sequence accuracy".title())
    ax2.grid(True, alpha=0.1)
    ax2.legend()
    
    ax3 = plt.subplot(223)
    ax3.plot(epoch_nums, train_first_n_acc, label="train" )
    ax3.plot(epoch_nums, dev_first_n_acc, label="dev")
    ax3.set_xlabel("Epoch Number")
    ax3.set_ylabel("first n-symbol accuracy".title())
    ax3.grid(True, alpha=0.1)
    ax3.legend()
    
    ax4 = plt.subplot(224)
    ax4.plot(epoch_nums, train_overlap_rate, label="train" )
    ax4.plot(epoch_nums, dev_overlap_rate, label="dev")
    ax4.set_xlabel("Epoch Number")
    ax4.set_ylabel("overlap rate".title())
    ax4.grid(True, alpha=0.1)
    ax4.legend()

    if saved_plot_fp != None:
        plt.savefig(saved_plot_fp, dpi=600, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_performances_per_seq_len(perfs):
    perfs = perfs.copy()
    perfs.pop("Aggregated")
    lengths = sorted([int(l.split("-")[-1]) for l in perfs.keys()])
    
    full_seq, first_n, overlap = [], [], []
    for l in lengths:
        full_seq.append(perfs[f"Len-{l}"]["full sequence accuracy"])
        first_n.append(perfs[f"Len-{l}"]["first n-symbol accuracy"])
        overlap.append(perfs[f"Len-{l}"]["overlap rate"])
    
    plt.figure(figsize=(len(lengths) // 4, len(lengths) // 10))
    plt.plot(lengths, full_seq, ".-", label="full sequence accuracy")
    plt.plot(lengths, first_n,  ".-", label="first n-symbol accuracy")
    plt.plot(lengths, overlap,  ".-", label="overlap rate")
    plt.xticks(lengths)
    plt.xlabel("Sequence Length")
    plt.ylabel("Performance")
    plt.grid(True, alpha=0.1)
    plt.legend()
    plt.show()
