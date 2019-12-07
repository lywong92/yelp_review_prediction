import gensim
import json
import pandas as pd
import nltk
import re
from sklearn.externals import joblib
import numpy as np
from collections import Counter
import logging
import math
nltk.download(quiet=True)

class LSTMModelInput:
    def __init__(self):
        self.wve_model = gensim.models.Word2Vec.load('./wve.model')
        self.cluster_data_file_name = './cluster_data.json'
        self.dataset_location = './normalised_data.json'
        self.cluster_data = None
        self.useful_data = None
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self._stop_words_ = nltk.corpus.stopwords.words()
        self.clustering = joblib.load('./cluster.model')
        self.train_test_data_file_path = './train_test_data.json'
        logging.basicConfig(level=logging.INFO)

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
            logging.info('Currently processing review %s:', review)
            no_tabs = str(review).replace('\t', ' ').replace('\n', '')
            alphas_only = re.sub("[^a-zA-Z\.]", " ", no_tabs)
            multi_spaces = re.sub(" +", " ", alphas_only)
            no_spaces = multi_spaces.strip()
            clean_text = no_spaces.lower()
            sentences = self.tokenizer.tokenize(clean_text)
            sentences = [re.sub("[\.]", "", sentence) for sentence in sentences]
            review_weights = range(self.cluster_data['num_clusters'])
            review_weights = dict(Counter(review_weights))
            review_weights = dict.fromkeys(review_weights, 0)

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
            final_weights = self.cluster_data['cluster_centers'] * sorted_review_weights[:, None]
            transformed_text.append(final_weights)
        
        self.input_data = np.array(transformed_text)
        self.labels = np.array(df['useful'].tolist())
        self.alternate_labels = np.array(df['useful_values'].tolist())

    def get_cluster_bucket(self, word):
        if word in self.wve_model.wv.vocab:
            word_embedding = self.wve_model[word]
            return self.clustering.predict([word_embedding])[0]
        else:
            return -1

    def save_training_data(self):
        rows, _ = self.useful_data.shape
        split_point = int(math.floor(rows * 0.7))
        train_input = self.input_data[0:split_point, :]
        test_input = self.input_data[split_point:rows, :]
        train_labels = self.labels[0:split_point]
        test_labels = self.labels[split_point:rows]
        train_alternate_labels = self.alternate_labels[0:split_point]
        test_alternate_labels = self.alternate_labels[split_point:rows]
        
        output_data = {}
        output_data['train_data'] = train_input.tolist()
        output_data['test_data'] = test_input.tolist()
        output_data['train_labels'] = train_labels.tolist()
        output_data['test_labels'] = test_labels.tolist()
        output_data['train_alternate_labels'] = train_alternate_labels.tolist()
        output_data['test_alternate_labels'] = test_alternate_labels.tolist()
        
        with open(self.train_test_data_file_path, 'w') as fh:
            json.dump(output_data, fh)
            fh.close()

model = LSTMModelInput()
model.load_data()
model.prepare_input()
model.save_training_data()