from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd
from ..Helpers import helpers
from ..Helpers.helpers import text_to_sentences, text_to_words, pad_words_from_both_sides
from tqdm.notebook import trange
import glob
import textstat

def cal_bert_score_chunk(scorer, target, source, chunk, batch_size=2, log_to_file=True, prefix="", folder="./bert_scores"):  
    import pickle as pkl
    import os
    if not os.path.exists(folder):
        os.mkdir(folder)
    P, R, F1 = scorer.score(target, source, batch_size=batch_size,verbose=False)

    pkl.dump((P.detach().cpu().numpy(),R.detach().cpu().numpy(),F1.detach().cpu().numpy()), open(f"{folder}/{prefix}_p_r_f1_{chunk}.pkl", "wb"))
    P = None
    R = None
    F1 = None

def clear_bert_scores(folder="./bert_scores"):
    import os
    files = glob.glob(f"{folder}/*p_r_f1_*.pkl")
    for file in files:
        os.remove(file)

def import_bert_scores(folder="./bert_scores", prefix=""):
    import glob
    import pickle as pkl
    scores = {}
    scores["f1"] = []
    scores["p"] = []
    scores["r"] = []
    
    files = glob.glob(f"{folder}/{prefix}*p_r_f1_*.pkl")
    for file in files:
        p = pkl.load(open(file,"rb"))
        scores["p"].extend(p[0])
        scores["r"].extend(p[1])
        scores["f1"].extend(p[2])

    scores_mean = {}
    scores_mean["f1"] = np.mean(scores["f1"])
    scores_mean["p"] = np.mean(scores["p"])
    scores_mean["r"] = np.mean(scores["r"])
    scores_var = {}
    scores_var["f1"] = np.std(scores["f1"])
    scores_var["p"] = np.std(scores["p"])
    scores_var["r"] = np.std(scores["r"])
    
    return scores_mean, scores_var, scores        
   

def cal_bert_score(target,source, batch_size = 2, chunks=5000, prefix="", folder="./bert_scores"):

    import glob
    import pickle as pkl
    import os
    import torch
    import gc
    from bert_score import BERTScorer
    scores = {}
    scores["f1"] = []
    scores["p"] = []
    scores["r"] = []
    
    #for i in tqdm(range(0,len(texts_basis), batch_size)):
    #    src_texts = texts_basis[i:i + batch_size]
       # print(src_texts[:2])
    #    summaries = texts_to_compare[i:i + batch_size]
        #print(summaries[:2])
    #    P, R, F1 = score(summaries, src_texts, lang='en', verbose=False, batch_size=batch_size, device="cuda:0")
     #   scores["f1"].extend(F1.numpy())
     #   scores["p"].extend(P.numpy())
      #  scores["r"].extend(R.numpy())
    
    #
    scorer = BERTScorer(lang="en", model_type="xlnet-base-cased", batch_size=batch_size, device="cuda:0")
    t = trange(0, len(target), chunks, desc=f"done in {chunks}")
    files = glob.glob(f"{folder}/*p_r_f1_*.pkl")
    for i in t:
        if f"{folder}/{prefix}_p_r_f1_{i}.pkl" not in files:
            t.set_description(f"done in {chunks}")
            cal_bert_score_chunk(scorer, target[i:i+chunks], source[i:i+chunks], i, batch_size, True, prefix, folder)
            gc.collect()
            torch.cuda.empty_cache()
        else:
            t.set_description(f"done in {chunks}, skipped: {i}")

    files = glob.glob(f"{folder}/*p_r_f1_*.pkl")
    #for file in files:
    #    p = pkl.load(open(file,"rb"))
    #    scores["p"].extend(p[0])
    #    scores["r"].extend(p[1])
    #    scores["f1"].extend(p[2])


    #scores_mean = {}
    #scores_mean["f1"] = np.mean(scores["f1"])
    #scores_mean["p"] = np.mean(scores["p"])
    #scores_mean["r"] = np.mean(scores["r"])
    #scores_var = {}
    #scores_var["f1"] = np.var(scores["f1"])
    #scores_var["p"] = np.var(scores["p"])
    #scores_var["r"] = np.var(scores["r"])


def cal_rouge(target, source, batch_size=64):
    from rouge import Rouge
    rouge_score = Rouge()
    scores = {}
    scores["r1-f1"] = []
    scores["r1-p"] = []
    scores["r1-r"] = []
    scores["r2-f1"] = []
    scores["r2-p"] = []
    scores["r2-r"] = []
    scores["rl-f1"] = []
    scores["rl-p"] = []
    scores["rl-r"] = []
    
    # Fix, so empty generations do not throw an error
    for l in range(len(target)):
        if len(target[l]) == 0:
            print(f"Target {l} was empty")
            target[l] = " "
    for i in tqdm(range(0,len(target), batch_size)):
        score = rouge_score.get_scores(target[i:i+batch_size], source[i:i+batch_size])
        scor_a = np.array(score)
        for l in scor_a:
            scores["r1-f1"].append(l["rouge-1"]["f"])
            scores["r1-p"].append(l["rouge-1"]["p"])
            scores["r1-r"].append(l["rouge-1"]["r"])
            scores["r2-f1"].append(l["rouge-2"]["f"])
            scores["r2-p"].append(l["rouge-2"]["p"])
            scores["r2-r"].append(l["rouge-2"]["r"])
            scores["rl-f1"].append(l["rouge-l"]["f"])
            scores["rl-p"].append(l["rouge-l"]["p"])
            scores["rl-r"].append(l["rouge-l"]["r"])
    
    scores_mean = {}
    scores_mean["r1-f1"] = np.mean(scores["r1-f1"])
    scores_mean["r1-p"] = np.mean(scores["r1-p"])
    scores_mean["r1-r"] = np.mean(scores["r1-r"])

    scores_mean["r2-f1"] = np.mean(scores["r2-f1"])
    scores_mean["r2-p"] = np.mean(scores["r2-p"])
    scores_mean["r2-r"] = np.mean(scores["r2-r"])

    scores_mean["rl-f1"] = np.mean(scores["rl-f1"])
    scores_mean["rl-p"] = np.mean(scores["rl-p"])
    scores_mean["rl-r"] = np.mean(scores["rl-r"])
    scores_std_dev = {}
    scores_std_dev["r1-f1"] = np.std(scores["r1-f1"])
    scores_std_dev["r1-p"] = np.std(scores["r1-p"])
    scores_std_dev["r1-r"] = np.std(scores["r1-r"])

    scores_std_dev["r2-f1"] = np.std(scores["r2-f1"])
    scores_std_dev["r2-p"] = np.std(scores["r2-p"])
    scores_std_dev["r2-r"] = np.std(scores["r2-r"])

    scores_std_dev["rl-f1"] = np.std(scores["rl-f1"])
    scores_std_dev["rl-p"] = np.std(scores["rl-p"])
    scores_std_dev["rl-r"] = np.std(scores["rl-r"])
    
    return scores_mean, scores_std_dev, scores

def cal_grammar_score(texts):
    import language_tool_python
    import numpy as np
    tool = language_tool_python.LanguageTool('en-US')
    scores_word_based = []
    scores_sentence_based = []
    for text in tqdm(texts):
        scores_word_based_sentence = []
        scores_sentence_based_sentence = []
        for sentence in helpers.text_to_sentences(text):
            matches = tool.check(sentence)
            count_errors = len(matches)
            # only check if the sentence is correct or not
            scores_sentence_based_sentence.append(np.min([count_errors, 1]))
            scores_word_based_sentence.append(count_errors)
        word_count = len(helpers.text_to_words(text))
        sum_count_errors_word_based = np.sum(scores_word_based_sentence)
        score_word_based = 1 - (sum_count_errors_word_based / word_count)
        scores_word_based.append(score_word_based)
        
        sentence_count = len(helpers.text_to_sentences(text))       
        sum_count_errors_sentence_based = np.sum(scores_sentence_based_sentence)
        score_sentence_based = 1 - np.sum(sum_count_errors_sentence_based / sentence_count)
        scores_sentence_based.append(score_sentence_based)

    return {"mean_score_sentences": np.mean(scores_sentence_based),\
            "stddev_score_sentences": np.std(scores_sentence_based), \
            "mean_score_words": np.mean(scores_word_based),\
            "var_score_words": np.mean(scores_word_based),\
            "scores_sentence_based": scores_sentence_based,
            "scores_word_based": scores_word_based}


def df_sentiments_review(sens):
    df = pd.concat([df_sentiment_review(sen) for sen in sens.values()])
    df_sen = df.drop("amount_timesteps",axis=1).groupby(level=0).agg("mean")
    df_sen["difference"] = abs(df_sen["means_orig"] - df_sen["means_gen"])
    return df_sen
def df_sentiment_review(sen):
    return pd.DataFrame.from_dict(data={"amount_timesteps":sen["amount_timesteps"], "means_orig": sen["means_orig"], "means_gen": sen["means_gen"] })
def df_sentiments_model(sens):
    df = pd.concat([df_sentiment_model(sen) for sen in sens.values()])
    return df.drop("amount_timesteps",axis=1).agg("mean")
def df_sentiment_model(sen):
    return pd.DataFrame(data={"amount_timesteps": [sen["amount_timesteps"]], "mae mean":[sen["mae-mean"]], "mae std dev":[sen["mae-mean-std-dev"]], "sum of mae-mean": [sen["Sum of mae-mean"]], "Sum of mae-std-dev": [sen["Sum of mae-std-dev"]] })


def rolling_windows(text, window_size = 5):
    splitted = text.split(" ")
    windows = []
    #print(f"going for: {len(splitted)} with {window_size}: {len(splitted) / window_size}")
    for i in range(0,len(splitted),window_size):
        windows.append(" ".join(splitted[i:i+window_size]))
    return windows

def classify_sentiment(classifier, text, max_window_size=5, amount_steps=1, expected_len=0):
    predicts=[]
    step_size = int(max_window_size / amount_steps)
    if step_size == 0:
        raise Exception(f"Step size may not be zero, this indicates that amount steps ({amount_steps}) was too large for max_window_size of ({max_window_size})")
    # +1 to make it run atleast one time, if amount_steps is 1
    for i in range(step_size, max_window_size + 1, step_size):
        predicts.extend(classifier([window for window in rolling_windows(text, i)]))
    
    if len(predicts) != expected_len and expected_len !=0:
        raise Exception(f"Length did not match: orig({expected_len} vs gen({len(predicts)})) text_was: {text_was}----------------------------------- text is: {text}")
    out = [0 if i["label"] == "NEGATIVE" else 1 for i in predicts ]
    return np.array(out)

def analyse_sentiment(result_dict, start_window_size=5, max_window_size=30):
    results = {}

    from transformers import pipeline

    # Allocate a pipeline for sentiment-analysis
    classifier = pipeline('sentiment-analysis', device=0)
    for amount_timesteps in tqdm(range(start_window_size, max_window_size, 1)):
        val_orig = []
        val_gen = []
        out_mae_vals = []
        for i in tqdm(range(len(result_dict["source_texts"])), leave=False):
            orig_text = result_dict["source_texts"][i]
            len_orig = len(text_to_words(orig_text))
            gen_text = result_dict["generated summaries"][i]
            len_gen = len(text_to_words(gen_text))
        # print(gen_text)
            #Add padding
            orig_text = pad_words_from_both_sides(orig_text, amount_timesteps)
            gen_text = pad_words_from_both_sides(gen_text, amount_timesteps)
            
            len_orig = len(text_to_words(orig_text))
            len_gen = len(text_to_words(gen_text))
            
            window_len_source = int(len_orig / amount_timesteps)
            window_len_gen = int(len_gen / amount_timesteps)
                
            orig_vals = np.array(classify_sentiment(classifier, orig_text, window_len_source, amount_steps=1))
            classified = np.array(list(classify_sentiment(classifier, gen_text, window_len_gen,amount_steps=1, expected_len=len(orig_vals))))

            val_orig.append(orig_vals)
            val_gen.append(classified)

            out_mae_vals.append(np.abs(orig_vals - classified))
            #orig_mean_b = [orig_vals[0]] * len(classified)
            #orig_var_b = [orig_vals[1]] * len(classified)
            #mse_means.append(((orig_mean_b - classified[:,0])**2).mean(axis=0))
            #mse_vars.append(((orig_var_b - classified[:,1])**2).mean(axis=0))
        out_means_mae = [np.mean(i) for i in out_mae_vals]
        out_std_dev_mae = [np.std(i) for i in out_mae_vals]
        mae_mean = np.mean(out_means_mae)
        mae_mean_std_dev = np.std(out_means_mae)

        results[amount_timesteps] = {"amount_timesteps": amount_timesteps, "means_orig": [np.mean(i) for i in val_orig], "means_gen": [np.mean(i) for i in val_gen], "mae-mean": mae_mean, "mae-mean-std-dev": mae_mean_std_dev, "Sum of mae-mean": np.round(np.sum(out_means_mae),4), "Sum of mae-std-dev": np.round(np.sum(out_std_dev_mae),4), "mae_functions": out_mae_vals, "functions_orig": val_orig, "functions_gen": val_gen}
    return results


def cal_compression_rate(count_orig_texts, count_sum_texts):
    import numpy as np
    compression_rates = [count_sum_texts[i] / count_orig_texts[i] for i in range(len(count_orig_texts))]
    return (np.mean(compression_rates), np.std(compression_rates))
    
def cal_generation_quality(target, source, tokenizer):
    
    t = target
    s = source

    orig_lens = [len(np.unique(i)) for i in tokenizer.texts_to_sequences(s)]
    gen_lens = [len(np.unique(i)) for i in tokenizer.texts_to_sequences(t)]

    div = [gen_lens[i] / orig_lens[i] for i in range(len(orig_lens))]
    return {"mean": np.mean(div), "std dev": np.std(div), "Div in %": div, "source lens": orig_lens, "target lens": gen_lens, "example_source": s[0], "example_gen": t[0]}


def cal_readability(target, source):
    import pandas as pd
    tf_r_es = [textstat.flesch_reading_ease(t) for t in target]
    tf_k_gs = [textstat.flesch_kincaid_grade(t) for t in target]
    td_c_rs = [textstat.dale_chall_readability_score(t) for t in target]
    
    sf_r_es = [textstat.flesch_reading_ease(t) for t in source]
    sf_k_gs = [textstat.flesch_kincaid_grade(t) for t in source]
    sd_c_rs = [textstat.dale_chall_readability_score(t) for t in source]
    
    diff_r_es = [np.abs(tf_r_es[i] - sf_r_es[i]) for i in range(len(tf_r_es))]
    diff_k_gs = [np.abs(tf_k_gs[i] - sf_k_gs[i]) for i in range(len(tf_k_gs))]
    difd_c_rs = [np.abs(td_c_rs[i] - sd_c_rs[i]) for i in range(len(td_c_rs))]
    
    return {"Flesch ease mean gen": np.mean(tf_r_es), \
            "Flesch ease mean orig": np.mean(sf_r_es), \
            "Flesch ease mean diff": np.mean(diff_r_es), \
            
            "Flesch grade mean gen": np.mean(tf_k_gs), \
            "Flesch grade mean orig": np.mean(sf_k_gs), \
            "Flesch grade mean diff": np.mean(diff_k_gs), \
            
            "Dale Chall Readability V2 mean gen": np.mean(td_c_rs), \
            "Dale Chall Readability V2 mean orig": np.mean(sd_c_rs), \
            "Dale Chall Readability V2 mean diff": np.mean(difd_c_rs), \
           },\
            \
            {"Flesch ease std dev gen": np.std(tf_r_es), \
            "Flesch ease std dev orig": np.std(sf_r_es), \
            "Flesch ease std dev diff": np.std(diff_r_es), \
            
            "Flesch grade std dev gen": np.std(tf_k_gs), \
            "Flesch grade std dev orig": np.std(sf_k_gs), \
            "Flesch grade std dev diff": np.std(diff_k_gs), \
            
            "Dale Chall Readability V2 std dev gen": np.std(td_c_rs),\
            "Dale Chall Readability V2 std dev orig": np.std(sd_c_rs),\
            "Dale Chall Readability V2 std dev diff": np.std(difd_c_rs)\
           }