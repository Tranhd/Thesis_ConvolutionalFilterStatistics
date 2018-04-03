import numpy as np
import sys
sys.path.append('../Thesis_CNN_mnist/')
from cnn import MnistCNN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

tf.reset_default_graph()
sess = tf.Session()
net = MnistCNN(sess, save_dir='../Thesis_CNN_mnist/Mnist_save/')
mnist = input_data.read_data_sets('MNIST_data/', reshape=False, one_hot=True)

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
                s1 = np.linalg.norm(pcas[k].transform(temp), ord=1, axis=0)
                s2 = np.mean(temp, axis=0)
                s3 = np.percentile(temp, [25, 50, 75])
            else:
                temp = np.reshape(layers[k][j], (-1, len(layers[k][j])))
                s1 = np.linalg.norm(pcas[k].transform(temp), ord=1)
                s2 = np.mean(temp)
                s3 = np.percentile(temp, [25, 50, 75])
            t = np.append(s1, s2)
            t = np.append(t, s3)
            layer.append(t)
        layer = np.asarray(layer)
        layersstats.append(layer)
    return layersstats



image = mnist.train.images[1:100,:,:,:]
labels = mnist.train.labels[1:100,:]
preds, _,_ = net.predict(image)
index = np.where(np.argmax(labels, 1) != preds)
print(len(index[0]))
image = np.concatenate((image[index[0]], image[0:len(index[0])]), axis=0)
print(image.shape)

layers = extract_activations(mnist.train.images[1:2000], net, batch_size=1000)
pcas = create_pca(layers)

layers = extract_activations(image, net, batch_size=2)
layersstats = create_layerstats(layers, pcas)
rows, cols = 1, 1

for stats in layersstats:
    fig, axes = plt.subplots(figsize=(20, 4), nrows=rows, ncols=cols, sharex=True, sharey=True, squeeze=True)
    print(stats.shape)
    axes.imshow(stats)
    plt.show()
plt.close()



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