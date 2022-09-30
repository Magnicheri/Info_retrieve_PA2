import code
import os
from collections import defaultdict, Counter
import pickle
import math
#import operator
#import nltk
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('stopwords')
#nltk.download('averaged_perceptron_tagger')
import stopwords as stopwords
from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datasets import load_dataset

class Indexer:
    def __init__(self):
        self.dbfile = "./ir.idx"
        self.tok2idx = defaultdict(lambda: len(self.tok2idx))
        self.idx2tok = dict()
        self.postings_lists = dict()
        self.docs = []
        self.raw_ds = []
        self.corpus_stats = 0
        if os.path.exists(self.dbfile):
            with open(self.dbfile, 'rb') as file:
                ds = pickle.load(file)
            self.tok2idx = ds['tok2idx']
            self.idx2tok = ds['idx2tok']
            self.docs = ds['docs']
            self.raw_ds = ds['raw_ds']
            self.postings_lists = ds['postings']
            self.corpus_stats = ds['avgdl']
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
        avg_dl = 0
        for docid, d in enumerate(tqdm(self.docs)):
            avg_dl += len(d)
            for i in d:
                if i in self.postings_lists:
                    if docid not in self.postings_lists[i][1]:
                        self.postings_lists[i][0] += 1
                    self.postings_lists[i][1].append(docid)
                else:
                    self.postings_lists[i] = [1,[docid]]
        self.corpus_stats = avg_dl/ len(self.raw_ds)
        print('Finalizing posting lists....')
        for k in tqdm(self.postings_lists):
            self.postings_lists[k][1] = Counter(self.postings_lists[k][1])
        index = {
                'avgdl' : self.corpus_stats,
                'tok2idx' : dict(self.tok2idx),
                'idx2tok' : self.idx2tok,
                'docs' : self.docs,
                'raw_ds' : self.raw_ds,
                'postings' : self.postings_lists
        }
        pickle.dump(index, open(self.dbfile, 'wb'))

class SearchAgent:
        k1 = 1.5                # BM25 parameter k1 for tf saturation
        b = 0.75                # BM25 parameter b for document length normalization
        def __init__(self, indexer):
            self.doc_id = []
            self.score = []
            self.doc_scores = []
            self.i = indexer

        def query(self, q_str):
            tokenizer = RegexpTokenizer(r"\w+")
            lemmatizer = WordNetLemmatizer()
            stop_words = stopwords.words('english')
            q_str = q_str.lower().strip()
            seq = tokenizer.tokenize(q_str)
            seq = [w for w in seq if w not in stop_words]
            seq = [lemmatizer.lemmatize(w) for w in seq]
            q_idx = seq
            for t in q_idx:
                if t in self.i.tok2idx:
                    index = self.i.tok2idx[t]
                    df = self.i.postings_lists[index][0]
                    tf = self.i.postings_lists[index][1]
                    avgdl = self.i.corpus_stats
                    for l in tf:
                        dl = len(self.i.docs[l])
                        real_dl = dl/avgdl
                        tfi = tf[l]
                        if l not in self.doc_id:
                            self.doc_id.append(l)
                            bmi25 = math.log2(len(self.i.docs)/df) * (((self.k1 + 1) * tfi ) / ((self.k1 * (1 - self.b + self.b * real_dl))+tfi ))
                            self.score.append(bmi25)
                        else:
                            idx = self.doc_id.index(l)
                            bmi25 = math.log2(len(self.i.docs) / df) * (((self.k1 + 1) * tfi) / ((self.k1 * (1 - self.b + self.b * real_dl)) + tfi))
                            self.score[idx] = self.score[idx] + bmi25
            self.doc_scores = list(zip(self.score,self.doc_id))
            self.doc_scores.sort(reverse=True)

            if len(self.doc_scores) == 0:
                return None
            else:
                self.display_results()

        def display_results(self):
        # Decode
            for docid_score in self.doc_scores[0:5]:  # print top 5 results
                print(f'\nDocID: {docid_score[1]}')
                print(f'Score: {docid_score[0]}')
                print('Article:')
                print(self.i.raw_ds[docid_score[1]])
            self.doc_id.clear()
            self.score.clear()
            self.doc_scores.clear()

if __name__ == "__main__":
    i = Indexer()
    q = SearchAgent(i)
    code.interact(local=dict(globals(), **locals()))
