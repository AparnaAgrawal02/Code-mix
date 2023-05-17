# %%
import numpy
# fine tune mt5 on dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
#import train split
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import torch
import torch.nn as nn
import klib
import os
import evaluate
import numpy as np
from nltk.tokenize import RegexpTokenizer
import argparse
parser = argparse.ArgumentParser(description='running model')
parser.add_argument('--model', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)
parser.add_argument("--text",type = str ,required=True)
args = parser.parse_args()
# %%
#tokenize
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
tokenizer.add_special_tokens({'additional_special_tokens': ['<sep>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['<pad>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['<s>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['</s>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['<unk>']})

maxlen = 512
def tokenize_df(df):
    target = tokenizer(df['sentence'], padding='max_length', truncation=True, return_tensors="pt", max_length=maxlen)
    input = tokenizer(df['english_translation'], padding='max_length', truncation=True, return_tensors="pt", max_length=maxlen)
    input_ids = input['input_ids']
    attention_mask = input['attention_mask']
    target_ids = target['input_ids']
    target_attention_mask = target['attention_mask']
    decoder_input_ids = target_ids.clone()
    #convert to tensors
    input_ids = torch.tensor(input_ids).squeeze()
    attention_mask = torch.tensor(attention_mask).squeeze()
    target_ids = torch.tensor(target_ids).squeeze()
    target_attention_mask = torch.tensor(target_attention_mask).squeeze()
   # decoder_input_ids = torch.tensor(decoder_input_ids)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': target_ids,
        #'decoder_input_ids': decoder_input_ids,
        #'decoder_attention_mask': target_attention_mask
    }



def tokenize_sentence(arg):
    encoded_arg =tokenizer(arg)
    return tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

      

# %%
model = MT5ForConditionalGeneration.from_pretrained(args.model)
#metrics = testing(model)

# %%
from torch.utils.data import DataLoader

# Predict with test data (first 5 rows)

input = args.text
input_ids = tokenizer(input, return_tensors="pt").input_ids
pred = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=1, no_repeat_ngram_size=2, max_length=128)
# Convert id tokens to text
text_preds = tokenizer.batch_decode(pred, skip_special_tokens=True)
#print(sentence_bleu(list(text_labels.split()),list(text_preds.split())))

# Show result
print("***** Input's Text *****")
print(input)
print("***** codemix (Generated Text) *****")
print(text_preds[0])



