import gensim
import json
import pandas as pd
import nltk
import sys
import re
from gensim.models import word2vec
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree
from sklearn.manifold import TSNE
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import logging
from sklearn.externals import joblib
nltk.download()

class ClusteringModel:
    def __init__(self, path):
        self.useful_data = None
        self.funny_data = None
        self.cool_data = None
        self.all_data = None
        self.dataset_location = path
        self.tokenized_sentences = None
        self.tokenized_data_path = './tokennized.txt'
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self._stop_words_ = nltk.corpus.stopwords.words()
        self.tsne_visualisation_file_name = './visualisation.png'
        self.wve_model_name = './wve.model'
        self.cluster_data_file_name = './cluster_data.json'
        self.cluster_model_file_name = './cluster.model'
        self.num_clusters = 120
        self.embedding_size = 250
        self.min_word_count = 50
        self.context = 30
        self.downsampling = 1e-2
        self.wve_model = None
        self.cluster_centers = None
        self.index = None
        self.centroid_map = None
        logging.basicConfig(level=logging.INFO)
    
    def read_data(self):
        logging.info("Reading data")
        with open(self.dataset_location, 'r') as fh:
            full_data = json.load(fh)
            self.useful_data = pd.DataFrame(json.loads(full_data['useful']))
            self.funny_data = pd.DataFrame(json.loads(full_data['funny']))
            self.cool_data = pd.DataFrame(json.loads(full_data['cool']))
            self.all_data = self.useful_data.append([self.funny_data, self.cool_data])
    
    def clean_data(self):
        logging.info("Cleaning data")
        with open(self.tokenized_data_path, 'w') as fh:
            reviews = self.all_data['text'].tolist()

            for index in range(len(reviews)):
                review = reviews[index]
                no_tabs = str(review).replace('\t', ' ').replace('\n', '')
                alphas_only = re.sub("[^a-zA-Z\.]", " ", no_tabs)
                multi_spaces = re.sub(" +", " ", alphas_only)
                no_spaces = multi_spaces.strip()
                clean_text = no_spaces.lower()
                sentences = self.tokenizer.tokenize(clean_text)
                sentences = [re.sub("[\.]", "", sentence) for sentence in sentences]

                if len(clean_text) > 0 and clean_text.count(' ') > 0:
                    for sentence in sentences:
                        sentence = sentence.split(' ')
                        pruned_sentence = [word for word in sentence if word not in self._stop_words_]
                        sentence = ' '.join(pruned_sentence)
                        if sentence: 
                            fh.write("%s\n" % sentence)

                if (index % 5000) == 0:
                    fh.flush()
            fh.close()
    
    def create_wve(self):
        logging.info("Creating word vectors")
        model = word2vec.Word2Vec( \
            word2vec.LineSentence(self.tokenized_data_path), \
            size=self.embedding_size, min_count = self.min_word_count, \
            window = self.context, sample = self.downsampling \
        )
        model.init_sims(replace=True)
        model.save(self.wve_model_name)
        self.wve_model = model

    def cluster_wv(self):
        logging.info("Clustering work vectors")
        clustering = KMeans(n_clusters = self.num_clusters, init='k-means++')
        index = clustering.fit_predict(self.wve_model.wv.syn0)
        joblib.dump(clustering, self.cluster_model_file_name)
        self.cluster_centers = clustering.cluster_centers_
        self.center_labels = [self.wve_model.most_similar(positive=[vector], topn=1)[0][0] for vector in self.cluster_centers]
        self.index = index
        self.centroid_map = dict(zip(self.wve_model.wv.index2word, index))

    def get_top_words(self):
        logging.info("Getting top words for each cluster")
        tree = KDTree(self.wve_model.wv.syn0)
        closest_points = [tree.query(np.reshape(x, (1, -1)), k=20) for x in self.cluster_centers]
        closest_words_ids = [x[1] for x in closest_points]
        closest_words = {}
        closest_ids = {}

        for i in range(0, len(closest_words_ids)):
            closest_words['Cluster #' + str(i)] = [self.wve_model.wv.index2word[j] for j in closest_words_ids[i][0]]
            closest_ids['Cluster #' + str(i)] = [self.wve_model[self.wve_model.wv.index2entity[j]] for j in closest_words_ids[i][0]]

        wd_df = pd.DataFrame(closest_words)
        id_df = pd.DataFrame(closest_ids)
        self.top_words = wd_df
        self.top_ids = id_df
    
    def display_cloud(self, cluster_index, color_map):
        logging.info("Printing word clouds for each cluster")
        if len(self.top_words['Cluster #' + str(cluster_index)])>0:
            wc = WordCloud(background_color="black", max_words=2000, max_font_size=80, colormap=color_map)
            wordcloud = wc.generate(' '.join([word for word in self.top_words['Cluster #' + str(cluster_index)]]))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.savefig('clusters/cluster_' + str(self.wve_model.most_similar(positive=[self.cluster_centers[cluster_index]], topn=1)[0][0]), bbox_inches='tight')
    
    def run_tsne_for_analysis(self):
        self.embedding_clusters = []
        for name, values in self.top_ids.iteritems():
            self.embedding_clusters.append(values)
        
        self.word_clusters = []
        for name, values in self.top_words.iteritems():
            self.word_clusters.append(values)

        self.embedding_clusters = np.array(self.embedding_clusters)
        x, y, z = self.embedding_clusters.shape
        tsne_model_en_2d = TSNE(perplexity=15, n_components=2, init='pca', n_iter=3500, random_state=32)
        self.embeddings_en_2d = np.array(tsne_model_en_2d.fit_transform(self.embedding_clusters.reshape(x * y, z))).reshape(x, y, 2)

    def tsne_plot_similar_words(self):
        plt.figure(figsize=(64, 36))
        colors = cm.rainbow(np.linspace(0, 1, len(self.center_labels)))

        for label, embeddings, words, color in zip(self.center_labels, self.embedding_clusters, self.word_clusters, colors):
            x = embeddings[:, 0]
            y = embeddings[:, 1]
            plt.scatter(x, y, c=color, alpha=0.7, label=label)
            for i, word in enumerate(words):
                plt.annotate(word, alpha=0.5, xy=(x[i], y[i]), xytext=(5, 2),
                            textcoords='offset points', ha='right', va='bottom', size=8)
        plt.legend(loc=4)
        plt.title('Yelp Data Visualisation')
        plt.grid(True)
        plt.savefig(self.tsne_visualisation_file_name, format='png', dpi=150, bbox_inches='tight')
        plt.clf()
    
    def save_cluster_centroids(self):
        logging.info("Saving cluster centroids")
        output = {}
        output['embedding_size'] = self.embedding_size
        output['num_clusters'] = self.num_clusters
        output['cluster_centers'] = self.cluster_centers.tolist()

        with open(self.cluster_data_file_name, 'w') as of_handle:
            json.dump(output, of_handle)
            of_handle.close()

dataset_folder = sys.argv[1]
clustering_model = ClusteringModel(dataset_folder)
clustering_model.read_data()
clustering_model.clean_data()
clustering_model.create_wve()
clustering_model.cluster_wv()
clustering_model.get_top_words()
clustering_model.run_tsne_for_analysis()
clustering_model.tsne_plot_similar_words()
clustering_model.save_cluster_centroids()

# cmaps = cycle([
#             'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
#             'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
#             'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])
# for i in range(150):
#     col = next(cmaps);
#     clustering_model.display_cloud(i, col)