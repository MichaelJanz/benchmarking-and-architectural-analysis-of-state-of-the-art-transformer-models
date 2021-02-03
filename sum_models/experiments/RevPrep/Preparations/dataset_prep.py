
import gzip
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from ..Helpers import helpers

def import_from_json(start_offset, amount, file_name):
    data = []
    val = 0
    with gzip.open(file_name) as fin:
        for line in tqdm(fin, total=start_offset + amount):
            if val <= start_offset:
                val = val + 1
                continue
            d = json.loads(line)
            data.append(d)
            val = val + 1
            if val == start_offset + amount:
                break
    return pd.DataFrame.from_dict(data)


def filter_and_rename_df(df, n_votes=4):
    df = df[df["n_votes"] >= n_votes]
    new_df = df[["book_id", "review_id", "rating", "review_text"]]
    new_df.columns = ["book_id", "review_id", "rating", "text"]
    return new_df


def prepare_text(text):
    txt = helpers.replace_escape(text)
    txt = helpers.replace_dbl_space(txt)
    txt = helpers.replace_special_char(txt)
    txt = helpers.replace_links(txt)
    return txt



def assign_word_counts(df):
    df["count_words"] = [len(helpers.text_to_words(x))
     for x in tqdm(df["text"])]
    return df


def selector_min_words(df, length, maximum):
    words_count = df["count_words"]
    selector = [(word_count > length and word_count < maximum)for word_count in words_count] 
    return selector


def selector_max_words(df, length):
    words_count = df["count_words"]
    selector = [word_count < length for word_count in words_count] 
    return selector


def filter_without_large_and_small_reviews(df, word_count_long = 900, word_count_short = 600, maximum_word_count = 1500):
    selector_long = selector_min_words(df, word_count_long, maximum_word_count)
    reviews_long = df[selector_long]
    selector = [x in reviews_long["book_id"].unique() for x in tqdm(df["book_id"], desc="1/2: Filtering by long sequences")]
    df_with_long_reviews = df[selector] 
    words_count_short = np.array([i for i in df_with_long_reviews["count_words"]])
    selector_short = selector_max_words(df_with_long_reviews, word_count_short)
    reviews_short = df_with_long_reviews[selector_short]

    unique = reviews_short["book_id"].unique()

    selector = [x in unique for x in tqdm(reviews_long["book_id"], desc="2/2: Filtering by short sequences")]
    reviews_long = reviews_long[selector]

    return reviews_long, reviews_short

    return df[selector]


def combine_short_long_reviews(reviews_long, reviews_short, min_length=400):
    long_summaries = []
    short_summaries = []
    for row in tqdm(reviews_long.iterrows(), total=len(reviews_long)):
        b_id = row[1]["book_id"]
        max_length = int(row[1]["count_words"] / 2)
        subset_short = reviews_short[[reviews_short["book_id"] == b_id][0]]
        subset_short = subset_short[subset_short["count_words"] <= max_length]
        subset_short = subset_short[subset_short["count_words"] > min_length]
        subset_short = subset_short[subset_short["rating"] == row[1]["rating"]]

        # The rest of the smaller summaries are good and can be used
        for subs_row in subset_short.iterrows():
            long_summaries.append(row[1]["text"])
            short_summaries.append(subs_row[1]["text"])

    return long_summaries, short_summaries


def filter_by_f1score(summaries_long, summaries_short, scores, threshold=0.85):
    summaries_long = np.array(summaries_long)
    summaries_short = np.array(summaries_short)  
    l_sums_filtered = summaries_long[np.array(scores["f1"]) > threshold]
    s_sums_filtered = summaries_short[np.array(scores["f1"]) > threshold]
    scores_filtered = np.array(scores["f1"])[np.array(scores["f1"]) > threshold]

    return l_sums_filtered, s_sums_filtered, scores_filtered

def filter_by_recall_score(summaries_long, summaries_short, scores, threshold=0.85):
    summaries_long = np.array(summaries_long)
    summaries_short = np.array(summaries_short)  
    l_sums_filtered = summaries_long[np.array(scores["r"]) > threshold]
    s_sums_filtered = summaries_short[np.array(scores["r"]) > threshold]
    scores_filtered = np.array(scores["r"])[np.array(scores["r"]) > threshold]

    return l_sums_filtered, s_sums_filtered, scores_filtered

def filter_by_score(summaries_long, summaries_short, score_index, scores, threshold):
    summaries_long = np.array(summaries_long)
    summaries_short = np.array(summaries_short)
    si = score_index  
    l_sums_filtered = summaries_long[np.array(scores[si]) > threshold]
    s_sums_filtered = summaries_short[np.array(scores[si]) > threshold]
    
    scores_filtered = {}
    for key in scores:
        scores_filtered[key] =  np.array(scores[key])[np.array(scores[si]) > threshold]

    return list(l_sums_filtered), list(s_sums_filtered), scores_filtered

def filter_by_shorts_grammarscore(summaries_long, summaries_short, scores, threshold_sentence=0.5, threshold_words = 0.94):
    s_sums_filtered = np.array(summaries_short)
    l_sums_filtered = np.array(summaries_long)
    s_sc_s = np.array(scores["scores_sentence_based"])
    s_sc_w = np.array(scores["scores_word_based"])
    s_sums_filtered_grammar = s_sums_filtered[(s_sc_s > threshold_sentence) & (s_sc_w > threshold_words)]
    l_sums_filtered_by_sg = l_sums_filtered[(s_sc_s > threshold_sentence) & (s_sc_w > threshold_words)]

    return l_sums_filtered_by_sg, s_sums_filtered_grammar

def filter_by_longs_grammarscore(summaries_long, summaries_short, scores, threshold_sentence=0.5, threshold_words = 0.94):
    s_sums_filtered = np.array(summaries_short)
    l_sums_filtered = np.array(summaries_long)
    l_sc_s = np.array(scores["scores_sentence_based"])
    l_sc_w = np.array(scores["scores_word_based"])
    l_sums_filtered_grammar = l_sums_filtered[(l_sc_s > threshold_sentence) & (l_sc_w > threshold_words)]
    s_sums_filtered_grammar = s_sums_filtered[(l_sc_s > threshold_sentence) & (l_sc_w > threshold_words)]

    return l_sums_filtered_grammar, s_sums_filtered_grammar


def filter_nonenglish(reviews_short, reviews_long):
    import langdetect
    from langdetect import DetectorFactory
    DetectorFactory.seed = 0

    reviews_short_eng = []
    reviews_long_eng = []

    for i in tqdm(range(len(reviews_long))):
        lang1 = langdetect.detect_langs(reviews_long[i])
        lang2 = langdetect.detect_langs(reviews_short[i])
        is_eng1=False
        is_eng2=False
        for l in lang1:
            if l.lang == "en" and l.prob > 0.8:
                is_eng1 = True
        for l in lang2:
            if l.lang == "en" and l.prob > 0.8:
                is_eng2 = True
        if is_eng1 and is_eng2:
            reviews_long_eng.append(reviews_long[i])
            reviews_short_eng.append(reviews_short[i])
    
    return reviews_short_eng, reviews_long_eng
