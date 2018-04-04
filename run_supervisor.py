import numpy as np
import sys
sys.path.append('../Thesis_CNN_mnist/')
from cnn import MnistCNN
sys.path.append('../Thesis_Utilities/')
from utilities import load_datasets
import tensorflow as tf
from filterstats_supervisor import Supervisor_filterstats


tf.reset_default_graph()
sess = tf.Session()
net = MnistCNN(sess, save_dir='../Thesis_CNN_mnist/Mnist_save/')

x_train, y_train, x_val, y_val, x_test, y_test = load_datasets(test_size=10000, val_size=5000, omniglot_bool=True,
                                                               name_data_set='data_omni_seed1337.h5', force=False,
                                                               create_file=True, r_seed=1337)


supervisor = Supervisor_filterstats(net, x_train, process_batch_size=5000)
normal_pool = x_train[25000:30000]
omni_pool = x_test[-500:]

supervisor.train(normal_pool, omni_pool)