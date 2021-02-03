from .pegasus import summarize_with_pegasus
from .bart import summarize_with_bart
from .t5 import summarize_with_T5


def generate(model_name, src_texts, model, tokenizer, window, window_size = 4, overlap=1, word_based=False, batch_size=8):    
    if "pegasus" in model_name:
        #its a pegasus model
        return summarize_with_pegasus(src_texts, model, tokenizer, window, window_size, overlap, batch_size ,word_based=word_based)
    
    elif "bart" in model_name:
        # its a bart-model
        return summarize_with_bart(src_texts, model, tokenizer, window, window_size, overlap, batch_size, word_based=word_based)
    else:
        # T5 or distilbart
        return summarize_with_T5(src_texts, model, tokenizer, window, window_size, overlap, batch_size, word_based=word_based)

def get_model_tokenizer(model_name):
    import torch
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if "pegasus" in model_name:
        #its a pegasus model
        from transformers import PegasusForConditionalGeneration, PegasusTokenizer
        tokenizer = PegasusTokenizer.from_pretrained(model_name)
        model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)    
        return model, tokenizer
    
    elif "bart-large" in model_name:
        # its a bart-model
        from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name).to(torch_device)
        return model, tokenizer

    elif "bart-custom-large" in model_name:
        from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name).to(torch_device)
        return model, tokenizer

    else:
        # T5 or distilbart
        from transformers import AutoTokenizer, AutoModelWithLMHead
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelWithLMHead.from_pretrained(model_name).to(torch_device)
        return model, tokenizer


def del_model_tokenizer(model,tokenizer):
    import torch
    del model
    del tokenizer
    torch.cuda.empty_cache()
        
        