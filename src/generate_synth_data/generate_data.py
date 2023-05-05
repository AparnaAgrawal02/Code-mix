import datasets
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from nltk.util import ngrams
import random
# import inltk
import nltk
# from inltk.inltk import setup
# from inltk.inltk import tokenize as tokenize_hi
from indicnlp.tokenize import indic_tokenize
from gensim.models import Word2Vec, KeyedVectors

print('libs loaded')


print('gettin dataset')
opus = datasets.load_dataset('opus100', 'en-hi', split='train', cache_dir='../data/OPUS/')


print('dataset downloaded')
class MixDataset(Dataset):
    def __init__(self, opus_data) :
        # self.tokenizer_en = get_tokenizer('basic_english')
        # self.tokenizer_hi = get_tokenizer('indic_tokenize', language='hi')
        self.opus_data = opus_data
        self.punc = set(['।', ',', '!', '(', ')', '–', ':', ';',
                            '?', '‘', '’', '“', '”', '॥', '‘', '’',
                            '-', '_', '{', '}', '[', ']', '<', '>',
                            '०', '!', '#', '$', '%', '&', '\\', '*',
                            '+', '-', '/', ':', ';', '=', '@', '^',
                            '_', '`', '|' '~'])
 
    
    def __len__(self):
        return len(self.opus_data)
    
    
    def __getitem__(self, idx):
        en, hi =  self.opus_data[idx]['translation']['en'], self.opus_data[idx]['translation']['hi']
        en = [word for word in nltk.word_tokenize(en) if word not in self.punc]
        hi = [word for word in nltk.word_tokenize(hi, language='hindi', preserve_line=True) if word not in self.punc]
        bi_grams_en = ngrams(en, 2)
        bi_grams_hi = ngrams(hi, 2)
        bi_en, bi_hi = [], []
        for w1, w2 in bi_grams_en :
            bi_en.append(w1 + '_' + w2)
        for w1, w2 in bi_grams_hi :
            bi_hi.append(w1 + '_' + w2)
        ret = en + hi + bi_en + bi_hi
        random.shuffle(ret)
        return ret
        

print('making ds, dl')
ds = MixDataset(opus)
dl = DataLoader(ds, batch_size=1, shuffle=True)


print('making model')
model = Word2Vec(sentences=dl, vector_size=100, window=5, min_count=1, workers=8)


print('training model')
model.train(dl, total_examples=len(opus), epochs=10)


print('saving wvs')
word_vectors = model.wv
word_vectors.save("../models/better_word2vec.wordvectors")