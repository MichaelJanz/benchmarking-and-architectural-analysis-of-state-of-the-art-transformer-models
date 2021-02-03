import numpy as np
import re
import glob
import pickle as pkl
import pandas as pd

def replace_escape(text):
    return text.replace("\'","")


# As recommended in Bert_Score github: https://github.com/Tiiiger/bert_score
def replace_dbl_space(text):
    return re.sub(r'\s+', ' ', text)


def replace_special_char(text):
    return re.sub('[^a-zA-Z0-9 \.\?\!;:,]', '', text)


def replace_links(text):
    return re.sub(r'http\S+', '', text)


def text_to_sentences(text):
    assert type(text) == type("") or type(text) == np.str_
    txt = text.replace("...", ".")
    #filter empty ones out
    return [ i for i in re.split("[\.\?\!]", txt) if i]


def text_to_words(text):
    return text.split(" ")


def cal_min_length(batch):
    val = int(np.mean([len(i) / 4 for i in batch["input_ids"]]))
    if val > 200:
        val = 200
    
    return val


def pad_words_from_both_sides(text, module_to=15):
    words = text_to_words(text)
    len_words = len(words)
    padding_needed = module_to - (len_words % module_to)
    padding = ""
    for i in range(int(padding_needed / 2)):
        padding = padding + " "
    text = padding + text
    padding = ""
    for i in range(int(padding_needed / 2), padding_needed,1):
        padding = padding + " "
    text = text + padding
    return text

def create_text_windows(text, window_size, overlap, word_based=False):
    import numpy as np
    windows = []
    if word_based:
        text_pieces = text_to_words(text)
    else:
        text_pieces = text_to_sentences(text)
    for i in range(0,len(text_pieces),window_size-overlap):
        start_index = i
        #Take upper border also
        if word_based:
            sentence_window = " ".join(text_pieces[start_index:start_index+window_size])
        else:
            sentence_window = ". ".join(text_pieces[start_index:start_index+window_size])
        windows.append(sentence_window)
    return windows

def get_long_label(label):
    if "mR" in label or "sR" in label:
        score_name = "Rouge"
        postfix1 = ""
        postfix2 = ""
        if "mR" in label:
            postfix2 = "mean"
        else:
            postfix2 = "std dev"
        if "-p" in label:
            postfix1 = "precision"
        elif "-r" in label:
            postfix1 = "recall"
        elif "-f1" in label:
            postfix1 = "F1-score"
        if "1" in label:
            return f"{score_name}-1 {postfix1} {postfix2}"
        if "2" in label:
            return f"{score_name}-2 {postfix1} {postfix2}"
        if "l" in label:
            return f"{score_name}-l {postfix1} {postfix2}"

    if "bs" in label:
        score_name = "Bertscore"
        postfix1 = ""
        postfix2 = ""
        if "mean" in label:
            postfix2 = "mean"
        else:
            postfix2 = "std dev"
        if "P" in label:
            postfix1 = "precision"
        elif "R" in label:
            postfix1 = "recall"
        elif "F1" in label:
            postfix1 = "F1-score"
        
        return f"{score_name} {postfix1} {postfix2}"

    if "Sen" in label:
        score_name = "Sentiment Analysis"
        postfix1 = ""
        postfix2 = ""
        if "diff" in label:
            postfix1 = "avg diff"
        elif "mae avg" in label:
            postfix1 = "mae avg"
        elif "mae std dev" in label:
            postfix1 = "mae std. dev."
        return f"{score_name} {postfix1}"
    
        

def get_short_name(name):
    if "pegasus-large" in name:
        return "pegasus-l"
    if "bart-large" in name:
        return "bart-lcnn"
    if "bart-finetuned" in name or "bart-custom" in name:
        return "bart-gr"
    if "t5-large" in name:
        return "t5-l"
    if "filtered" in name:
        return "filtered goodreads"
    return "cnndm"



def sort_samples(data_frames):
    import numpy as np
    import pandas as pd
    vals = list(data_frames.values())
    keys = list(data_frames.keys())
    keys_same = [k[0:-6] for k in keys]
    indexes_same = [k[-5:-4] for k in keys]
    
    # Get the unique ids for each entry, so it can be sorted into 1,2 or 3
    same_keys = np.unique(keys_same, return_inverse=True)
    
    dfs = {}
    for i in range(len(keys_same)):
        if keys_same[i] not in dfs:
            dfs[keys_same[i]] = []
        if type(vals[i]) == pd.DataFrame:
            df = list(vals)[i]
            df.index = [rp(f"{df.index[0]}_S{indexes_same[i]}", sample=True)]
        else: #Its a dict for the result_windows
            df = pd.DataFrame(data={"duration_total": vals[i]["time"], "compression rate mean": vals[i]["compression_rate_mean"], "compression rate std dev": vals[i]["compression_rate_std_dev"]}, index=[rp(f"{keys_same[i]}_S{indexes_same[i]}", False)])

        dfs[keys_same[i]].append(df)

    data_ = {}
    for k in dfs.keys():
        data_[rp(k, sample=False)] = pd.concat(dfs[k])
        
    return data_


def load_result_files(folder, add_regex=""):
    files = glob.glob(f"{folder}/*{add_regex}.pkl")
    data = {}
    for f in files:
        data[f] = pkl.load(open(f,"rb"))
    
    return data


def rp(model_name, sample=True):
    if sample:
        return model_name.replace(".pkl", "").replace("_sample_",", sample: ").replace("bart-custom-large_finetuned/best_tfmr/", "bart-finetuned-goodreads").replace("googlepegasus","google/pegasus").replace("facebookbart","facebook/bart").replace("pseudo","filtered goodreads").replace("./","")
    else:
        return model_name.replace(".pkl", "").replace("_sample_S","").replace("_S","").replace("1","").replace("2","").replace("3","").replace("googlepegasus","google/pegasus").replace("facebookbart","facebook/bart").replace("bart-custom-large_finetuned/best_tfmr/", "bart-finetuned-goodreads").replace("pseudo","filtered goodreads").replace("./","").replace("benchmark_results/bertscore/","").replace("benchmark_results/sentimentanalysis/","").replace("_sample","")


def df_mean_samples(vals, keys, use_postfix=True):
    dfs = []
    for k in vals.values():
        if use_postfix:
            df = pd.DataFrame(data={f"{keys[0].replace('mean','')} mean":k[keys[0]].aggregate("mean"), f"{keys[1].replace('stddev','').replace('std dev','')} std dev": k[keys[1]].aggregate("mean")}, index=[rp(k.index[0],False)])
        else:
            df = pd.DataFrame(data={f"{keys[0].replace('mean','')}":k[keys[0]].aggregate("mean"), f"{keys[1].replace('stddev','').replace('std dev','')}": k[keys[1]].aggregate("mean"),f"{keys[2].replace('stddev','').replace('std dev','')}": k[keys[2]].aggregate("mean") }, index=[rp(k.index[0],False)])
        dfs.append(df)

    if use_postfix:
        return pd.concat(dfs).sort_values(f"{keys[0].replace('mean','')} mean", ascending=False)
    else:
       return pd.concat(dfs).sort_values(f"{keys[0].replace('mean','')}", ascending=False)

    
def concat_mean_std(vals_mean, vals_std_dev):
    vals = {}
    for i in range(len(vals_mean.keys())):
        km = list(vals_mean.keys())[i]
        ks = list(vals_std_dev.keys())[i]

        vm = vals_mean[km]
        vs = vals_std_dev[ks]

        df = pd.concat([vm, vs],axis=1)
        vals[rp(df.index[0], False)] = df
    return vals