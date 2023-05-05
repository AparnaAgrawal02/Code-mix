import datasets
from nltk.util import ngrams
from torch.utils.data import DataLoader, Dataset
# import inltk
import nltk
# from inltk.inltk import setup
# from inltk.inltk import tokenize as tokenize_hi
from indicnlp.tokenize import indic_tokenize
from gensim.models import Word2Vec, KeyedVectors
import unicodedata
import time
import sys
import json
import multiprocessing


print('libs loaded')

class CM_Dataset(Dataset):
    def __init__(self, path='/home2/shreya.patil/Courses/NLP/Code-mix/models/better_word2vec.wordvectors'):
        self.path = path
        self.word_vectors = self.load_wvs(path)
        self.opus = datasets.load_dataset('opus100', 'en-hi', split='train', cache_dir='/home2/shreya.patil/Courses/NLP/Code-mix/data/OPUS/')
        self.length = len(self.opus)
        self.punc = set(['।', ',', '!', '(', ')', '–', ':', ';',
                            '?', '‘', '’', '“', '”', '॥', '‘', '’',
                            '-', '_', '{', '}', '[', ']', '<', '>',
                            '०', '!', '#', '$', '%', '&', '\\', '*',
                            '+', '-', '/', ':', ';', '=', '@', '^',
                            '_', '`', '|' '~'])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Retrieve the data for a particular index and process it here
        sent = self.opus[idx]['translation']['en']
        final_sent = ' '.join(self.modify_sent(sent, 3))
        return (idx, {'en': sent, 'cm': final_sent})
    
    
    def load_wvs(self, path='/home2/shreya.patil/Courses/NLP/Code-mix/models/better_word2vec.wordvectors'):
        return KeyedVectors.load(path, mmap='r')
    
    def is_hindi(self, word) :
        for char in word:
            if unicodedata.name(char).startswith('DEVANAGARI'):
                return True
            return False

    def is_english(self, word):
        for char in word:
            if unicodedata.name(char).startswith('LATIN'):
                return True
            return False
    

    def most_similar(self, word, topn=5):
        return self.word_vectors.most_similar(word, topn=topn)


    def get_closest_hindi(self, en_word) :
        query = self.word_vectors.get_vector((en_word,))
        sim_words = self.most_similar(query)
        for word in sim_words :
            try :
                if (self.is_hindi(word[0][0])) :
                    return word
            except :
                return (('',), 0.0)
        
        return (('',), 0.0)
    

    def split_ngram(self, n_gram) :
        return n_gram.split('_')



    def modify_sent(self, sent, num_replacements) :
        sent = [word for word in nltk.word_tokenize(sent) if word not in self.punc]
        bi_grams_pre = ngrams(sent, 2)
        bi_grams = []
        for w1, w2 in bi_grams_pre :
            bi_grams.append(w1 + '_' + w2)
        # print(sent)
        unigram_cws = [(i, 1, self.get_closest_hindi(token)) for i, token in enumerate(sent)]
        bigram_cws = [(i, 2, self.get_closest_hindi(token)) for i, token in enumerate(bi_grams)]
        cws = unigram_cws + bigram_cws
        
        # [print(unigram_cws)]
        argsorted_indices = sorted(range(len(cws)), key=lambda i: cws[i][2][1], reverse=True)
        
        replacements = {}
        count = 0
        for idx in argsorted_indices :
            token = cws[idx]
            if (token[0] not in replacements) :
                replacements[token[0]] = token[2][0][0]
                if (token[1] == 2) : replacements[token[0] + 1] = None
                count += 1 
            if count == num_replacements : break

        ret = []
        for i in range(len(sent)) :
            if i in replacements :
                if (replacements[i] is None) : continue
                [ret.append(word) for word in self.split_ngram(replacements[i])]
                # ret.pop(replacements[i])
                
            
            else :
                ret.append(sent[i])
        
        return ret



ds = CM_Dataset()
dl = DataLoader(ds, batch_size=8, num_workers=4)

t0 = time.time()
for i in range(2) :
    print(next(iter(dl)))
    

print(time.time() - t0)