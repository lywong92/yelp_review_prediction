import gensim
import json
import pandas as pd
import nltk
import re
from sklearn.externals import joblib
import numpy as np
from collections import Counter
nltk.download(quiet=True)

class LSTMModel:
    def __init__(self):
        self.wve_model = gensim.models.Word2Vec.load('./wve.model')
        self.cluster_data_file_name = './cluster_data.json'
        self.dataset_location = './'
        self.cluster_data = None
        self.useful_data = None
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self._stop_words_ = nltk.corpus.stopwords.words()
        self.clustering = joblib.load('./cluster.model')

    def load_data(self):
        with open(self.cluster_data_file_name) as fh:
            self.cluster_data = json.load(fh)
            fh.close()

        with open(self.dataset_location, 'r') as fh:
            full_data = json.load(fh)
            self.useful_data = pd.DataFrame(json.loads(full_data['useful']))
            fh.close()

    def prepare_input(self):
        df = self.useful_data.sample(frac=1).reset_index(drop=True)
        text_input = df['text'].tolist()
        transformed_text = []

        for _, review in enumerate(text_input):
            no_tabs = str(review).replace('\t', ' ').replace('\n', '')
            alphas_only = re.sub("[^a-zA-Z\.]", " ", no_tabs)
            multi_spaces = re.sub(" +", " ", alphas_only)
            no_spaces = multi_spaces.strip()
            clean_text = no_spaces.lower()
            sentences = self.tokenizer.tokenize(clean_text)
            sentences = [re.sub("[\.]", "", sentence) for sentence in sentences]
            review_weights = range(self.cluster_data['embedding_size'])
            review_weights = dict(Counter(review_weights))

            if len(clean_text) > 0 and clean_text.count(' ') > 0:
                for sentence in sentences:
                    sentence = sentence.split(' ')
                    pruned_sentence = [self.get_cluster_bucket(word) for word in sentence if word not in self._stop_words_]

                    if len(pruned_sentence) > 0:
                        pruned_sentence = dict(Counter(pruned_sentence))
                        for key in review_weights:
                            if key in pruned_sentence:
                                review_weights[key] = review_weights[key] + pruned_sentence[key]
            
            sorted_review_weights = np.array([review_weights[key] for key in sorted(review_weights.keys())])
            sorted_review_weights = sorted_review_weights/np.sum(sorted_review_weights)
            final_weights = self.cluster_data['cluster_centers'] * sorted_review_weights[:, None]
            transformed_text.append(final_weights)

    def get_cluster_bucket(self, word):
        word_embedding = self.wve_model[word]
        return self.clustering.predict(word_embedding)[0]

model = LSTMModel()
model.load_data()
model.prepare_input()