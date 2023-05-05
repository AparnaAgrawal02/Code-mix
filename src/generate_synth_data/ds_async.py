import datasets
from nltk.util import ngrams
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
from indic_transliteration import sanscript
# import multiprocessing


print('libs loaded')


# sample_dict = {0 : (0, 44526) ,
# 1 : (44526, 89052) ,
# 2 : (89052, 133578) ,
# 3 : (133578, 178104) ,
# 4 : (178104, 222630) ,
# 5 : (222630, 267156) ,
# 6 : (267156, 311682) ,
# 7 : (311682, 356208) ,
# 8 : (356208, 400734) ,
# 9 : (400734, 445260) ,
# 10 : (445260, 489786) ,
# 11 : (489786, 534312)}

# Arguments passed
# script_index = int(sys.argv[1])
# index_range = sample_dict[script_index]
# n_elements = index_range[1] - index_range[0]
# print("Script Index: ", script_index, "\tRange: ", index_range, "\tn_elements: ", n_elements, flush=True)



def load_wvs(path='/home2/shreya.patil/Courses/NLP/Code-mix/models/word2vec.wordvectors'):
    return KeyedVectors.load(path, mmap='r')




def is_hindi(word) :
    for char in word:
        if unicodedata.name(char).startswith('DEVANAGARI'):
            return True
    return False




def is_english(word):
    for char in word:
        if unicodedata.name(char).startswith('LATIN'):
            return True
    return False



def most_similar(word, word_vectors, topn=5):
    return word_vectors.most_similar(word, topn=topn)



def get_closest_hindi(en_word, word_vectors) :
    query = word_vectors.get_vector((en_word,))
    sim_words = most_similar(query, word_vectors)
    for word in sim_words :
        try :
            if (is_hindi(word[0][0])) :
                return word
        except :
            return (('',), 0.0)
    
    return (('',), 0.0)




def split_ngram(n_gram) :
    return n_gram.split('_')



def modify_sent(sent, num_replacements, word_vectors, punc) :
    sent = [word for word in nltk.word_tokenize(sent) if word not in punc]
    bi_grams_pre = ngrams(sent, 2)
    bi_grams = []
    for w1, w2 in bi_grams_pre :
        bi_grams.append(w1 + '_' + w2)
    # print(sent)
    unigram_cws = [(i, 1, get_closest_hindi(token, word_vectors)) for i, token in enumerate(sent)]
    bigram_cws = [(i, 2, get_closest_hindi(token, word_vectors)) for i, token in enumerate(bi_grams)]
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

    
    input_script = sanscript.DEVANAGARI
    output_script = sanscript.ITRANS
    ret = []
    for i in range(len(sent)) :
        if i in replacements :
            if (replacements[i] is None) : continue
            [ret.append(sanscript.transliterate(word, input_script, output_script)) for word in split_ngram(replacements[i])]
            # ret.pop(replacements[i])
            
        
        else :
            ret.append(sent[i])
    
    return ret




print("------------Imports Completed-------------")

sample_dict = {0 : (0, 44526) ,
            1 : (44526, 89052) ,
            2 : (89052, 133578) ,
            3 : (133578, 178104) ,
            4 : (178104, 222630) ,
            5 : (222630, 267156) ,
            6 : (267156, 311682) ,
            7 : (311682, 356208) ,
            8 : (356208, 400734) ,
            9 : (400734, 445260) ,
            10 : (445260, 489786) ,
            11 : (489786, 534312)}


# Arguments passed
script_index = int(sys.argv[1])
index_range = sample_dict[script_index]
n_elements = index_range[1] - index_range[0]
print("Script Index: ", script_index, "\tRange: ", index_range, "\tn_elements: ", n_elements)


punc = set(['।', ',', '!', '(', ')', '–', ':', ';',
                        '?', '‘', '’', '“', '”', '॥', '‘', '’',
                        '-', '_', '{', '}', '[', ']', '<', '>',
                        '०', '!', '#', '$', '%', '&', '\\', '*',
                        '+', '-', '/', ':', ';', '=', '@', '^',
                        '_', '`', '|' '~'])



print('("------------gettin dataset("------------', flush=True)
opus = datasets.load_dataset('opus100', 'en-hi', split='train', cache_dir='/home2/shreya.patil/Courses/NLP/Code-mix/data/OPUS/')


print('("------------dataset downloaded"------------', flush=True)

        
word_vectors = load_wvs('/home2/shreya.patil/Courses/NLP/Code-mix/models/better_word2vec.wordvectors')

sentences = []
t0 = time.time()
for i in range(index_range[0], index_range[1]) :
    if(i % 25 == 0) :
        t1 = time.time()
        report = f"Progress: {((i-index_range[0])/n_elements)*100} | i: {i} | Run Time: {t1 - t0}"
        print(report, flush=True)
    
    
    sent = opus[i]['translation']['en']
    if len(sent) > 150 : continue
    final_sent = ' '.join(modify_sent(sent, 3, word_vectors, punc))
    sentences.append({'en': sent, 'cm': final_sent})


print("Writing to JSON", flush=True)
with open(f'/home2/shreya.patil/Courses/NLP/Code-mix/models/gen_data/sentences_{script_index}.json', 'w') as file:
    json.dump(sentences, file)

print(f"{script_index}_Data Stored", flush=True)



