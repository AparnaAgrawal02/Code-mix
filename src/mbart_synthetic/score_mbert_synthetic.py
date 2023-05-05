# %%
import numpy
# fine tune mt5 on dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import  BartForConditionalGeneration
#import train split
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import torch
import torch.nn as nn
from google.transliteration import transliterate_word
import klib
import os
#bleu score
from torchtext.data.metrics import bleu_score
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, BertTokenizer, BertModel
# %%
#tokenize
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
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




# %%
train = load_dataset('csv', data_files='../train.csv')
val = load_dataset('csv', data_files='../val.csv')
test = load_dataset('csv', data_files='../test.csv')
train = train.map(tokenize_df, batched=True, batch_size=128,remove_columns=['sentence','english_translation'])
val = val.map(tokenize_df, batched=True, batch_size=128,remove_columns=['sentence','english_translation'])
test = test.map(tokenize_df, batched=True, batch_size=128,remove_columns=['sentence','english_translation'])


# %%
import evaluate
import numpy as np
from nltk.tokenize import RegexpTokenizer


bleu_metric = evaluate.load("sacrebleu")

rouge_metric = evaluate.load("rouge")

def tokenize_sentence(arg):
    encoded_arg =tokenizer(arg)
    return tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)

def metrics_func(eval_arg):
    preds, labels = eval_arg
    # Replace -100
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Convert id tokens to text
    text_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    text_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Insert a line break (\n) in each sentence for ROUGE scoring
    # (Note : Please change this code, when you perform on other languages except for Japanese)
    text_preds = [(p if p.endswith(("!", "！", "?", "？", "。")) else p + "。") for p in text_preds]
    text_labels = [(l if l.endswith(("!", "！", "?", "？", "。")) else l + "。") for l in text_labels]
    sent_tokenizer_jp = RegexpTokenizer(u'[^!！?？。]*[!！?？。]')
    text_preds = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(p))) for p in text_preds]
    text_labels = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(l))) for l in text_labels]
    # compute ROUGE score with custom tokenization
    #blue score
    texts_bleu =[text.strip() for text in text_preds]
    labels_bleu = [[text.strip()] for text in text_labels]
    result = bleu_metric.compute(predictions=texts_bleu, references=labels_bleu)
    return rouge_metric.compute(
        predictions=text_preds,
        references=text_labels,
        tokenizer=tokenize_sentence
    ), result['score']

# %%
from torch.utils.data import DataLoader
#tokenizer = MT5Tokenizer.from_pretrained("./mt5")
def testing(model):
    metrics =[]
    sample_dataloader = DataLoader(
      test["train"].with_format("torch"),
      collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model),
      batch_size=5)
    for batch in sample_dataloader:
      with torch.no_grad():
        preds = model.generate(
          batch["input_ids"],
          num_beams=15,
          num_return_sequences=1,
          no_repeat_ngram_size=1,
          remove_invalid_values=True,
          max_length=128,
        )
      labels = batch["labels"]
      metric = metrics_func([preds, labels])
      metrics.append(metric)
    return metrics

def average_metric(metrics):
    rouge = 0
    rouge2 = 0
    rougeL = 0
    rougeLsum = 0
    bleu = 0
    for metric in metrics:
        rouge += metric[0]['rouge1']
        rouge2 += metric[0]['rouge2']
        rougeL += metric[0]['rougeL']
        rougeLsum += metric[0]['rougeLsum']
        bleu += metric[1]
    return rouge/len(metrics),rouge2/len(metrics),rougeL/len(metrics),rougeLsum/len(metrics),bleu/len(metrics)
      

# %%

model = BartForConditionalGeneration.from_pretrained("/ssd_scratch/cvit/aparna/mbert-synthetic")

metrics = testing(model)

# %%
print("mbert_with_real_data")
scores = average_metric(metrics)
print("rouge:",scores[0])
print("rouge2:",scores[1])
print("rougeL:",scores[2])
print("rougeLsum:",scores[3])
print("bleu:",scores[4])


# %%
from torch.utils.data import DataLoader

# Predict with test data (first 5 rows)
sample_dataloader = DataLoader(
  test["train"].with_format("torch"),
  collate_fn=DataCollatorForSeq2Seq(tokenizer, model=model),
  batch_size=5)
for batch in sample_dataloader:
  with torch.no_grad():
    preds = model.generate(
      batch["input_ids"],
      num_beams=15,
      num_return_sequences=1,
      no_repeat_ngram_size=1,
      remove_invalid_values=True,
      max_length=128,
    )
  labels = batch["labels"]
  inputs = batch["input_ids"]
  break

# Replace -100 (see above)
inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

# Convert id tokens to text
text_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
text_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
text_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)
#print(bleu_score(list(text_labels.split()),list(text_preds.split())))

# Show result
print("***** Input's Text *****")
print(text_inputs[2])
print("***** codemix (True Value) *****")
print(text_labels[2])
print("***** codemix (Generated Text) *****")
print(text_preds[2])

# %%
for i in range(5):
    print("***** Input's Text *****")
    print(text_inputs[i])
    print("***** codemix (True Value) *****")
    print(text_labels[i])
    print("***** codemix (Generated Text) *****")
    print(text_preds[i])


