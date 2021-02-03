from ...Helpers.helpers import text_to_sentences, text_to_words, cal_min_length, create_text_windows
from ...Evaluations.scoring import cal_compression_rate
from tqdm.notebook import tqdm

def summarize_with_T5(src_texts, model, tokenizer, window=False, window_size=4, overlap=1, batch_size=4, word_based=False):
    import torch
    import time
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not window:
        start = time.time()
        out_texts = []
        inputs = tokenizer(src_texts, return_tensors='pt').to(torch_device)
        summary_ids = model.generate(inputs['input_ids'], min_length=cal_min_length(inputs), early_stopping=True).to(torch_device)
        out_texts = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
        return {"source_texts": src_texts, "generated summaries": out_texts, "time": time.time() - start}
    else:
        count_words_gen = []
        count_words_orig = []
        out_texts = []
        start = time.time()
        for text in tqdm(src_texts, desc="Texts done", leave=False):
            count_words_orig.append(len(text_to_words(text)))
            count_gen_temp = 0
            out_temp = []
            sentence_windows = create_text_windows(text, window_size, overlap, word_based)
            sentence_windows = [f"summarize: {i}" for i in sentence_windows]
            generated_batches = []

            for i in range(0, len(sentence_windows), batch_size):
                sentence_window = sentence_windows[i:i+batch_size]
                inputs = tokenizer(sentence_window, truncation=True, padding='longest', return_tensors='pt').to(torch_device)
                min_length = cal_min_length(inputs)
                # Batch is too small, when having less than 20 tokens
                if min_length <= 20:
                    continue
                summary_ids = model.generate(inputs['input_ids'], min_length=cal_min_length(inputs), max_length = cal_min_length(inputs) *3, early_stopping=True).to(torch_device)
                ids_before = inputs['input_ids']
                generated_text = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
                # Put all texts together
                generated_batches.extend(generated_text)
                del generated_text
                del summary_ids
                del inputs
                torch.cuda.empty_cache()
            generated_text = " ".join(generated_batches) 
            count_gen_temp = len(text_to_words(generated_text))
            count_words_gen.append(count_gen_temp)
            out_texts.append(generated_text)
        torch.cuda.empty_cache()
        comp_rate_mean, comp_rate_std_dev = cal_compression_rate(count_words_orig, count_words_gen)
        return {"source_texts": src_texts, "generated summaries": out_texts, "time": time.time() - start, "count_words_orig": count_words_orig, "count_words_gen":count_words_gen, "compression_rate_mean": comp_rate_mean, "compression_rate_std_dev": comp_rate_std_dev}
