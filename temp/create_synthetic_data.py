
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
# for indiclanguage
from indicnlp.tokenize import indic_tokenize
import random
from indicnlp.tokenize import indic_detokenize


# import opus100 dataset
from datasets import load_dataset
dataset = load_dataset('opus100', 'en-hi', split='train')

print(dataset[0])
#ngrams (n,xi,yi) is set of unique n-grams in xi and yi
def cummulative_ngrams(dataset ):
    created_texts = []
    ngrams ={}
    f = open("synthetic_data.txt", "w")
    for r in range(len(dataset)):
        # tokenize english and hindi
        english = dataset['translation'][r]['en']
        english = indic_tokenize.trivial_tokenize(english)
        #print(english)
        hindi = dataset['translation'][r]['hi']
        hindi = indic_tokenize.trivial_tokenize(hindi,lang='hi')
        #print(hindi)
        for n in range(1,4):
            ngram = (n,dataset['translation'][r]['en'],dataset['translation'][r]['hi'])
            if (n==1):
                 ngrams[ngram] = []
            else:
                ngrams[ngram] = ngrams[(n-1,dataset['translation'][r]['en'],dataset['translation'][r]['hi'])][::]
            for i in range(len(english)-n):
                ngrams[ngram].append(" ".join([i+"_" for i in english[i:i+n]])[:-1])
            for j in range(len(hindi)-n):
                ngrams[ngram].append(" ".join([i+"_" for i in hindi[i:i+n]])[:-1])
            if n==3:
                cum_list = ngrams[ngram]
                #suffle cum_list
                random.shuffle(cum_list)
                #print(cum_list)
                #created_texts.append(" ".join(cum_list))
                
                f.write(" ".join(cum_list)+"\n")
    #return created_texts

                    



