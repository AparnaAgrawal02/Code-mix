# %%
import numpy
# fine tune mt5 on dataset
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, BertTokenizer, BertModel
from transformers import Trainer, TrainingArguments
from transformers import BertGenerationEncoder,BertGenerationDecoder,EncoderDecoderModel
from datasets import load_dataset
from transformers import  BartForConditionalGeneration,MBartTokenizer,MBartForConditionalGeneration
from transformers import XLMTokenizer, XLMForSequenceClassification

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
torch.cuda.is_available()

# %%


# %%
#load dataset
#English-Hindi code-mixed parallel corpus.csv
df = pd.read_json("../sentences_0.json")
df = df.dropna()
df = df.reset_index(drop=True)

df.head()


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

#print lens
print(len(train))
print(len(val))
print(len(test))

#save train, val, test
train.to_csv('train1.csv', index=False)
val.to_csv('val1.csv', index=False)
test.to_csv('test1.csv', index=False)


# %%
df.columns

# %%
#tokenize
tokenizer =  MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")

# Load pre-trained XLM model
#tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
tokenizer.add_special_tokens({'additional_special_tokens': ['<sep>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['<pad>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['<s>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['</s>']})
tokenizer.add_special_tokens({'additional_special_tokens': ['<unk>']})





# %%
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


# %%
#tokenize train, val, test
train = load_dataset('csv', data_files='train1.csv')
val = load_dataset('csv', data_files='val1.csv')
test = load_dataset('csv', data_files='test1.csv')
train = train.map(tokenize_df, batched=True, batch_size=128,remove_columns=['en','cm'])
val = val.map(tokenize_df, batched=True, batch_size=128,remove_columns=['en','cm'])
test = test.map(tokenize_df, batched=True, batch_size=128,remove_columns=['en','cm'])


# %%
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

#model = BertGenerationEncoder.from_pretrained("bert-base-multilingual-cased")


# encoder = BertGenerationEncoder.from_pretrained("bert-base-multilingual-cased", bos_token_id=101, eos_token_id=102)
# # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
# decoder = BertGenerationDecoder.from_pretrained(
#     "bert-large-uncased", add_cross_attention=True, is_decoder=True, bos_token_id=101, eos_token_id=102
# )
# model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model =  MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25").to(device)


#model = BartForConditionalGeneration.from_pretrained("mbert-codemixed/checkpoint-1500").to(device)

# model.encoder.resize_token_embeddings(len(tokenizer))
# model.decoder.resize_token_embeddings(len(tokenizer))
model.resize_token_embeddings(len(tokenizer))


# %%


#training args


training_args = Seq2SeqTrainingArguments(
  output_dir = "/ssd_scratch/cvit/aparna/mbart-synthetic-codemixed",
  log_level = "error",
  num_train_epochs = 10,
  learning_rate = 5e-5,
  lr_scheduler_type = "linear",
  warmup_steps = 90,
  optim = "adafactor",
  weight_decay = 0.01,
  per_device_train_batch_size =1,
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
trainer.save_model("/ssd_scratch/cvit/aparna/mbart")
