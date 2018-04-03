import numpy as np
import sys
sys.path.append('../Thesis_CNN_mnist/')
from cnn import MnistCNN
sys.path.append('../Thesis_Utilities/')
from utilities import load_datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import tensorflow as tf
from sklearn import svm
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import metrics

tf.reset_default_graph()
sess = tf.Session()
net = MnistCNN(sess, save_dir='../Thesis_CNN_mnist/MnistCNN_save/')

x_train, y_train, x_val, y_val, x_test, y_test = load_datasets(test_size=10000, val_size=5000, omniglot_bool=True,
                                                               name_data_set='data_omni_seed1337.h5', force=False,
                                                               create_file=True, r_seed=1337)

def extract_activations(train_images, net, batch_size=5000):
    layers = []
    for i in range(len(train_images)//batch_size):
        _, _, activations = net.predict(train_images[i*batch_size:i*batch_size+batch_size])
        for k in range(len(activations)):
            if i == 0:
                layers.append(activations[k])
            else:
                layers[k] = np.concatenate((layers[k], activations[k]), axis=0)
    return layers

def create_pca(layers):
    pcas = []
    for layer in layers:
        shape = layer.shape
        Zm = layer.reshape((-1,shape[-1]))
        pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=shape[-1]))])
        pipeline.fit_transform(Zm)
        pcas.append(pipeline)
    return pcas

def create_layerstats(layers, pcas):
    layersstats = []
    for k in range(len(layers)):
        layer = []
        for j in range(len(layers[k])):
            if len(layers[k][j].shape) > 2:
                temp = np.reshape(layers[k][j], (-1,layers[k][j].shape[-1]))
                s1 = 1/temp.shape[0] * np.linalg.norm(pcas[k].transform(temp), ord=1, axis=0)
                s2 = np.percentile(temp, [25, 50, 75], axis=0)
                s3 = np.max(temp, axis=0)
            else:
                temp = np.reshape(layers[k][j], (-1, len(layers[k][j])))
                s1 = pcas[k].transform(temp)
                s2 = np.percentile(temp, [25, 50, 75])
                s3 = np.max(temp)
            t = np.append(s1, s2)
            t = np.append(t, s3)
            layer.append(t)
        layer = np.asarray(layer)
        layersstats.append(layer)
    return layersstats



image1 = x_test[-20:,:,:,:]
image2 = x_test[0:20,:,:,:]
image = np.concatenate((image1, image2), axis=0)
print(image.shape)
layers = extract_activations(x_train[:2000], net, batch_size=2000)
pcas = create_pca(layers)

normal_pool = x_train[20001:30000]
omnipool = x_test[-1000:]

print(x_test.shape)
test_x = x_test[9000:11100]
layers_test = extract_activations(test_x, net, batch_size=1000)
layersstats_test = create_layerstats(layers_test, pcas)
test_y = np.concatenate((np.ones((1000)),np.zeros((1000))), axis=0)

layers_normal = extract_activations(normal_pool, net, batch_size=1000)
layersstats_normal = create_layerstats(layers_normal, pcas)

layers_omni = extract_activations(omnipool, net, batch_size=1000)
layersstats_omni = create_layerstats(layers_omni, pcas)

for ln, lo, lt in zip(layersstats_normal, layersstats_omni, layersstats_test):
    normal_sample = ln[np.random.randint(len(ln), size=len(lo)),:]
    T = np.concatenate((normal_sample, lo))
    print(T.shape)
    labels = np.concatenate((np.ones((len(lo))),np.zeros((len(lo)))))
    print(labels.shape)
    idx = np.random.permutation(len(T))
    T = T[idx]
    labels = labels[idx]
    clf = svm.SVC()
    clf.fit(T, labels)
    score = clf.decision_function(lt)
    fpr, tpr, threshold = metrics.roc_curve(test_y, score)
    i = np.abs(tpr - 0.97).argmin()
    th = threshold[i]
    classification = (score >= th)
    print()
    print(f'Accuracy: {np.sum(classification == test_y)/len(test_y)}')


"""
rows, cols = 1, 1

for stats in layersstats:
    print(stats.shape)
    part = int((stats.shape[-1] - 3)/3)
    print(part)
    fig, axes = plt.subplots(figsize=(10, 4), nrows=1, ncols=4, sharex=True, sharey=True, squeeze=True)
    for i,ax in enumerate(axes):
        if i != 3:
            ax.imshow(stats[:,i*part:i*part + part])
        else:
            ax.imshow(stats[:,-3:])
    plt.show()
plt.close()
"""

"""
w = 2
h = 2
n_pixels = h*w
size_data = 500
n_features = 4
single_example = np.random.randint(-1,1,size=(n_features*n_pixels, n_features))
Zm = np.random.randint(-1,1,size=(size_data*n_features*n_pixels, n_features))


pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=n_features))])
pipeline.fit_transform(Zm)
zmi = pipeline.transform(single_example)
print(np.linalg.norm(zmi, ord=1, axis=0).shape)
print(zmi.shape)
print(zmi)
"""