# %%
import numpy
# fine tune mt5 on dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from simpletransformers.t5 import T5Model, T5Args
from transformers import pipeline
#import train split
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.preprocessing
import torch
import torch.nn as nn
from google.transliteration import transliterate_word
import klib
import os

# %%
#load dataset
#English-Hindi code-mixed parallel corpus.csv
df = pd.read_json('sentences_0.json')
df = df.dropna()
df = df.reset_index(drop=True)
# add column for prefix
df['prefix'] = 'translate English to Hinglish: '



# %%
#data cleaning 

df=klib.data_cleaning(df)

# %%
#split train, val, test
# convert df  so that it can be used by transformers


train, test = train_test_split(df, test_size=0.2, random_state=42)
train, val = train_test_split(train, test_size=0.2, random_state=42)
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)



#save train, val, test
train.to_csv('train1.csv', index=False)
val.to_csv('val1.csv', index=False)
test.to_csv('test1.csv', index=False)


#tokenize
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
tokenizer.add_special_tokens({'additional_special_tokens': ['<sep>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['<pad>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['<s>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['</s>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['<unk>']})


maxlen = 512
def tokenize_df(df):
    target = tokenizer([str(i) for i in df["cm"]], padding='max_length', truncation=True, return_tensors="pt", max_length=maxlen)
    input = tokenizer([str(i) for i in df["en"]], padding='max_length', truncation=True, return_tensors="pt", max_length=maxlen)
    #print("x")
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


#tokenize train, val, test
train = load_dataset('csv', data_files='train1.csv')
val = load_dataset('csv', data_files='val1.csv')
test = load_dataset('csv', data_files='test1.csv')
train = train.map(tokenize_df, batched=True, batch_size=128,remove_columns=['en','cm'])
val = val.map(tokenize_df, batched=True, batch_size=128,remove_columns=['en','cm'])
test = test.map(tokenize_df, batched=True, batch_size=128,remove_columns=['en','cm'])



import evaluate
import numpy as np
from nltk.tokenize import RegexpTokenizer

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
    return rouge_metric.compute(
        predictions=text_preds,
        references=text_labels,
        tokenizer=tokenize_sentence
    )

# %%
# finetuen mt5
os.environ["WANDB_DISABLED"] = "true"
model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
model.resize_token_embeddings(len(tokenizer))

#training args


training_args = Seq2SeqTrainingArguments(
  output_dir = "mt5-synthetic",
  log_level = "error",
  num_train_epochs = 10,
  learning_rate = 5e-4,
  lr_scheduler_type = "linear",
  warmup_steps = 90,
  optim = "adafactor",
  weight_decay = 0.01,
  per_device_train_batch_size = 2,
  per_device_eval_batch_size = 1,
  gradient_accumulation_steps = 16,
  evaluation_strategy = "steps",
  eval_steps = 100,
  predict_with_generate=True,
  generation_max_length = 128,
  save_steps = 500,
  logging_steps = 10,
  push_to_hub = False
)


#trainer
trainer = Seq2SeqTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train["train"],         # training dataset
    eval_dataset=val["train"],             # evaluation dataset
    tokenizer=tokenizer,               # tokenizer
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model), # data collator
    
)

#train
trainer.train()

#save model
trainer.save_model("./mt5_synthetic")

# %%
from torch.utils.data import DataLoader
model = MT5ForConditionalGeneration.from_pretrained("./mt5_synthetic")
#tokenizer = MT5Tokenizer.from_pretrained("./mt5")


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
  break
print(preds, labels)
metrics_func([preds, labels])

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

# %%



