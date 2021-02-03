from ...Helpers.helpers import text_to_sentences, text_to_words, cal_min_length, create_text_windows
from ...Evaluations.scoring import cal_compression_rate
from tqdm.notebook import tqdm


def summarize_with_pegasus(src_texts, model, tokenizer, window, window_size=4, overlap=1, batch_size=2, word_based=False):
    import time
    import torch
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not window:
        start = time.time()
        out_texts = []
        for i in range(0, len(src_texts), batch_size):
            texts = src_texts[i, i+batch_size]
            batch = tokenizer.prepare_seq2seq_batch(texts, truncation=True, padding='longest').to(torch_device)
            translated = model.generate(**batch, min_length=cal_min_length(batch))
            tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for text in tgt_text:
                out_texts.append(tgt_text)
        return {"source_texts": src_text, "generated summaries": out_texts, "time": time.time() - start}
    else:
        count_words_gen = []
        count_words_orig = []
        out_texts = []
        start = time.time()
        for text in tqdm(src_texts, desc="Texts done", leave=False):
            count_words_orig.append(len(text_to_words(text)))
            count_gen_temp = 0
            sentence_windows = create_text_windows(text, window_size, overlap, word_based)
            generated_batches = []

            for i in range(0, len(sentence_windows), batch_size):
                sentence_window = sentence_windows[i:i+batch_size]
                batch = tokenizer.prepare_seq2seq_batch(sentence_window, truncation=True, padding='longest').to(torch_device)
                min_length = cal_min_length(batch)
                # Batch is too small, when having less than 20 tokens
                if min_length <= 20:
                    continue
                translated = model.generate(**batch, min_length=min_length, max_length=min_length *3, early_stopping=True).to(torch_device)
                generated_text = tokenizer.batch_decode(translated, skip_special_tokens=True) #all as using batch_size > 1
                # Put all texts together
                generated_batches.extend(generated_text)
                del generated_text
                del translated
                del batch
                torch.cuda.empty_cache()
            generated_text = " ".join(generated_batches)
            count_gen_temp = len(text_to_words(generated_text))
            count_words_gen.append(count_gen_temp)
            out_texts.append(generated_text)
        torch.cuda.empty_cache()
        comp_rate_mean, comp_rate_std_dev = cal_compression_rate(count_words_orig, count_words_gen)
        return {"source_texts": src_texts, "generated summaries": out_texts, "time": time.time() - start, "count_words_orig": count_words_orig, "count_words_gen":count_words_gen, "compression_rate_mean": comp_rate_mean, "compression_rate_std_dev": comp_rate_std_dev}
