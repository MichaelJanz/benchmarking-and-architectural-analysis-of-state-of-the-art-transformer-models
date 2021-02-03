from matplotlib import pyplot as plt
import numpy as np
import matplotlib.cm as cm
from ..Helpers.helpers import get_long_label, get_short_name
from itertools import cycle

def f1scores(scores):
    size = len(scores["f1"])
    ran = range(size)
    plt.plot(ran, scores["f1"])
    plt.plot(ran, [np.mean(scores["f1"])] * size, color="green", label="mean")
    plt.plot(ran, [np.median(scores["f1"])] * size, color="blue", label="median")
    plt.legend()
    plt.show()

def rscores(scores):
    size = len(scores["r"])
    ran = range(size)
    plt.scatter(ran, scores["r"], alpha=0.4, s=1**2)
    plt.plot(ran, [np.mean(scores["r"])] * size, color="green", label="mean")
    plt.plot(ran, [np.median(scores["r"])] * size, color="blue", label="median")
    plt.legend()
    plt.show()

def pscores(scores):
    size = len(scores["p"])
    ran = range(size)
    plt.scatter(ran, scores["p"], alpha=0.4, s=1**2)
    plt.plot(ran, [np.mean(scores["p"])] * size, color="green", label="mean")
    plt.plot(ran, [np.median(scores["p"])] * size, color="blue", label="median")
    plt.legend()
    plt.show()

def scores(scores, key, limit, label="", title=""):
    size = len(scores[key])
    ran = range(size)
    plt.scatter(ran, scores[key], alpha=0.4, s=1**2)
    plt.plot(ran, [np.mean(scores[key])] * size, color="green", label="mean")
    plt.plot(ran, [np.median(scores[key])] * size, color="blue", label="median", linestyle=":")
    plt.plot(ran, [limit] * size, color="red", label="limit")
    plt.xlabel("Samples")
    plt.ylabel(label)
    plt.title(title)
    plt.legend()
    plt.show()

def grammar_scores(scores, limit_words, limit_sentences, title):
    
    l = len(scores["scores_word_based"])
    plt.scatter(range(l), scores["scores_word_based"], label="scores_word_based", alpha=0.4, s=1**2)
    plt.scatter(range(l), scores["scores_sentence_based"], label="scores_sentence_based", alpha=0.4,s=1**2)
    plt.plot(range(l), [np.mean(scores["scores_word_based"])] * l, color="green", label="mean words", linestyle="--")
    plt.plot(range(l), [np.median(scores["scores_word_based"])] * l, color="blue", label="median words", linestyle="--")
    plt.plot(range(l), [np.mean(scores["scores_sentence_based"])] * l, color="lime", label="mean sentence", linestyle="--")
    plt.plot(range(l), [np.median(scores["scores_sentence_based"])] * l, color="cyan", label="median sentence", linestyle="--")
    plt.plot(range(l), [limit_words] * l, color="red", label="limit words")
    plt.plot(range(l), [limit_sentences] * l, color="purple", label="limit sentences")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Grammar Scores")
    plt.show()


def plot_sentiment_mae(analysis_results, first_n=100):
    fig = plt.figure(figsize=(20, 4))
    for a in range(len(analysis_results)):
        analysis_result = analysis_results[a]
        ax = fig.add_subplot(1, len(analysis_results),a + 1)
        ax.set_title(f"MAE of orig vs gen, ts={analysis_result['amount_timesteps']}")
        functions = analysis_result["mae_functions"][:first_n]
        for i in range(len(functions)):
            ax.plot(range(len(functions[i])),functions[i], label=i)
        ax.legend()
    
def plot_sentiment_functions(analysis_results, first_n=100):
    fig = plt.figure(figsize=(20, 4))
    for a in range(len(analysis_results)):
        analysis_result = analysis_results[a]
        f_orig = analysis_result["functions_orig"][:first_n]
        f_gen = analysis_result["functions_gen"][:first_n]
        NUM_COLORS = len(f_orig)
        cm = plt.get_cmap('gist_rainbow')
        ax = fig.add_subplot(1, len(analysis_results),a + 1)
        ax.set_title(f"Sentiment orig vs gen, ts={analysis_result['amount_timesteps']}")
        for i in range(len(f_orig)):
            lines = ax.plot(range(len(f_orig[i])), f_orig[i], linestyle="dashed", label=f"orig: {i}" )
            lines[0].set_color(cm(i/NUM_COLORS))
            lines = ax.plot(range(len(f_gen[i])), f_gen[i], label=f"gen: {i}")
            lines[0].set_color(cm(i/NUM_COLORS))
        ax.legend()
        
def plot_sentiment_means(analysis_results, first_n = 100):
    fig = plt.figure(figsize=(20, 4))
    for a in range(len(analysis_results)):
        analysis_result = analysis_results[a]
        ax = fig.add_subplot(1, len(analysis_results),a + 1)
        ax.set_title(f"Means of MAE of orig vs gen, ts={analysis_result['amount_timesteps']}")
        means_orig = analysis_result["means_orig"][:first_n]
        means_gen = analysis_result["means_gen"][:first_n]
        ax.scatter(range(len(means_orig)), means_orig, linestyle="dashed", label="Means orig", alpha=0.4, s=1**2)
        ax.scatter(range(len(means_gen)), means_gen, label="Means gen", alpha=0.4, s=1**2)
        ax.legend()

def plot_result_values(result_vals, fields, errorbar_fields=None, s_len=3, increase_size=False, alpha_dot=0.4, alpha_line=0.85, alpha_err=0.75, size_dot=1**2, size_line=2, size_err=2, capsize=9):
    
    if increase_size:
        fig = plt.figure(figsize=(30, 5))
    else:
        fig = plt.figure(figsize=(20, 4))

    lines = ["--","-.",":"]
    linecycler = cycle(lines)
    for f_i in range(len(fields)):
        ax = fig.add_subplot(1, len(result_vals),f_i+1)
        colors = plt.get_cmap('gist_rainbow')
        field = fields[f_i]
        for i in range(len(result_vals)):
            #c = cm.rainbow(np.linspace(0.3+i*0.4, 0.3+i*0.4, 3),4)
            c = [colors(0.15+(1/(len(result_vals)+0.15)*i)) for l in range(s_len)]
            ls = next(linecycler)
            vals = list(result_vals.values())[i]
            name = list(result_vals.keys())[i]
            lbl = get_short_name(name)
            #for l in range(3):
            ax.scatter(range(len(vals[field])),vals[field], color=c, label=lbl, alpha=alpha_dot, s=size_dot)
            if errorbar_fields is not None:
                err_vals = vals[errorbar_fields[f_i]]
                eb = ax.errorbar(range(len(vals)),vals[field], fmt=".", yerr=err_vals, color=c[0], alpha=alpha_err, capsize=capsize, capthick=size_err)
                eb[-1][0].set_linestyle(ls)
            mean = np.mean(vals[field])
            ax.plot([0,s_len-1],[mean] * 2, c=c[0],linestyle=ls, label=f"{lbl}-avg", linewidth=size_line, alpha=alpha_line)

        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xticks(range(s_len))
        ax.set_xticklabels([f"sample {s+1}" for s in range(s_len)], rotation=30)
        ax.set_title(get_long_label(field))
        fig.tight_layout()