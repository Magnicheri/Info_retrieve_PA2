
import os
from collections import defaultdict, Counter
import pickle
import math
import operator
import nltk
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
import stopwords as stopwords

from tqdm import tqdm
from nltk import pos_tag, WhitespaceTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datasets import load_dataset

class Indexer:
    global dbfile
    dbfile = "./ir.idx"

    def __init__(self):
        self.tok2idx = defaultdict(lambda: len(self.tok2idx))
        self.idx2tok = dict()
        self.postings_lists = dict()
        self.docs = []
        self.rawds = []
        self.corpus_stats = {'avgdl': 0}
        if os.path.exists(dbfile):
            #with open('dbfile.pkl', 'rb') as file:
             #ds = pickle.load(file)
            pass
        else:
         ds = load_dataset("cnn_dailymail", '3.0.0', split="test")
         self.raw_ds = ds['article']
         self.clean_text(self.raw_ds)
         self.create_postings_lists()
         #with open('dbfile.pkl', 'wb') as file:
           # pickle.dump(self.postings_lists, file)

    def clean_text(self, lst_text, query=False):
        tokenizer = RegexpTokenizer(r"\w+")
        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words('english')
        for l in lst_text:
            encoded_doc = []
            l = l.lower().strip()
            seq = tokenizer.tokenize(l)
            seq = [w for w in seq if w not in stop_words]
            seq = [lemmatizer.lemmatize(w) for w in seq]
            for w in seq:
                self.idx2tok[self.tok2idx[w]] = w
                encoded_doc.append(self.tok2idx[w])
            self.docs.append(encoded_doc)

    def create_postings_lists(self):

        for docid, d in enumerate(self.docs):
            self.corpus_stats['avgdl'] += len(d)
            for i in d:
                if i in self.postings_lists:
                    self.postings_lists[i][0] += 1
                    if docid not in self.postings_lists[i][1]:
                        self.postings_lists[i][1].append(docid)
                else:
                    self.postings_lists[i] = [1,[docid]]
        print('Hello')
        self.corpus_stats['avgdl'] /= len(self.docs)

class SearchAgent:
        k1 = 1.5                # BM25 parameter k1 for tf saturation
        b = 0.75                # BM25 parameter b for document length normalization

        def __init__(self, indexer):
            self.i = indexer

        def query(self, q_str):

            q_idx = self.i.clean_text([q_str])
            for t in q_idx:
                df = self.i.posting_lists[t][0]
               # w = math.log2((len(self.i.docs) - df + .5) / (df + .5)) * (
                            #tfi * (self.k1 + 1) / (tfi + self.k1(1 - self.b + self.b(dl / avgdl))))
            results = {}
            if len(results) == 0:
                return None
            else:
                self.display_results(results)


        def display_results(self, results):
        # Decode
            for docid, score in results[:5]:  # print top 5 results
                print(f'\nDocID: {docid}')
                print(f'Score: {score}')
                print('Article:')
                print(self.i.raw_ds[docid])



i = Indexer()
#q = SearchAgent(i)
print(i)