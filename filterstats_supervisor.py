from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import svm
import numpy as np
from sklearn import metrics
from sklearn.externals import joblib
import os
import glob


class Supervisor_filterstats(object):

    def __init__(self, net, train_images, process_batch_size=5000, force=False):
        self.net = net
        if not force and os.path.isdir('./pca'):
            print('log: Using existing pca transforms')
            self.filenames = glob.glob('./pca/*.pkl')
        else:
            if not os.path.isdir('./pca'): os.mkdir('./pca')
            layers = self.get_activations(train_images, process_batch_size)
            print('log: Activations extracted')
            self.filenames = self.make_pcatransforms(layers)
            print('log: Pca transforms created and saved')
        self.classifiers = [svm.SVC() for _ in range(len(self.filenames))]


    def get_activations(self, train_images, batch_size):
        layers = []
        for i in range(len(train_images) // batch_size):
            _, _, activations = self.net.predict(train_images[i * batch_size:i * batch_size + batch_size])
            for k in range(len(activations)):
                if i == 0:
                    layers.append(activations[k])
                else:
                    layers[k] = np.concatenate((layers[k],
                                                     activations[k]), axis=0)
        return layers

    def make_pcatransforms(self, layers):
        filenames = [f'./pca/pca{k}.joblib.pkl' for k in range(len(layers))]
        for filename, layer in zip(filenames, layers):
            shape = layer.shape
            Zm = layer.reshape((-1, shape[-1]))
            pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=shape[-1]))])
            pipeline.fit_transform(Zm)
            _ = joblib.dump(pipeline, filename)
        return filenames

    def get_stats(self, images, batch_size=1000):
        layers = self.get_activations(images, batch_size)
        layersstats = []
        for k in range(len(layers)):
            layer = []
            pca = joblib.load(self.filenames[k])
            for j in range(len(layers[k])):
                if len(layers[k][j].shape) > 2:
                    temp = np.reshape(layers[k][j], (-1, layers[k][j].shape[-1]))
                    s1 = 1 / temp.shape[0] * np.linalg.norm(pca.transform(temp), ord=1, axis=0)
                    s2 = np.percentile(temp, [25, 50, 75], axis=0)
                    s3 = np.max(temp, axis=0)
                else:
                    temp = np.reshape(layers[k][j], (-1, len(layers[k][j])))
                    s1 = pca.transform(temp)
                    s2 = np.percentile(temp, [25, 50, 75])
                    s3 = np.max(temp)
                t = np.append(s1, s2)
                t = np.append(t, s3)
                layer.append(t)
            layer = np.asarray(layer)
            layersstats.append(layer)
        return layersstats

    def run_classifier(self, index, normal_stats, omni_stats, tp=0.97):
        if len(normal_stats) >= len(omni_stats):
            idx = np.arange(len(normal_stats))
            idx = np.random.permutation(idx)
            sample_normal = normal_stats[idx[:len(omni_stats)]]
        else:
            sample_normal = normal_stats

        T = np.concatenate((sample_normal, omni_stats))
        labels = np.concatenate((np.ones((len(sample_normal))), np.zeros((len(omni_stats)))))
        idx = np.random.permutation(len(T))
        T = T[idx]
        labels = labels[idx]
        self.classifiers[index].fit(T, labels)
        score = self.classifiers[index].decision_function(normal_stats)
        fpr, tpr, threshold = metrics.roc_curve(np.ones((len(normal_stats))), score, pos_label=1)
        i = np.abs(tpr - tp).argmin()
        th = threshold[i]
        classification = (score >= th)
        return classification

    def train(self, n_pool, o_pool):
        n_stats = self.get_stats(n_pool, batch_size=5000)
        o_stats = self.get_stats(o_pool, batch_size=500)
        current_layer = 0
        total_layers = len(n_stats)
        normal_examples_left = np.ones((len(n_stats[0])), dtype=bool)
        while current_layer <= total_layers or normal_examples_left.sum() > 0:
            classified_as_normal = self.run_classifier(current_layer, n_stats[current_layer][normal_examples_left],
                                                       o_stats[current_layer], tp=0.97)
            normal_examples_left[classified_as_normal] = 0
            current_layer = (current_layer+1) % total_layers
            print(normal_examples_left.sum())

