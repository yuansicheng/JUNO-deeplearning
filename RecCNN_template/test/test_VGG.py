#coding:utf-8
#author: Ziyuan Li, Zhu jiang, Yue Huang, Zhen Qian

import os, sys
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # log level
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # select a GPU card

sys.path.append("..")
from RecCNN import par
from RecCNN.VGG16 import VGG16
from RecCNN.data import get_dataset

FLAGS = par.FLAGS
FLAGS.modelname = os.path.splitext(os.path.basename(__file__))[0]
FLAGS.modelname = FLAGS.modelname.replace("test","train")
par.init_test_par()

if not os.path.exists(par.path_to_result):
    os.makedirs(par.path_to_result)


_, test_db = get_dataset(par.path_to_tfr, testdataset=True)

modelfile = par.path_to_model + '.hdf5'


def main():
    
    #select a model
    network = VGG16()
    modelfile = par.path_to_model + '.hdf5'
    network.load_weights(modelfile)

    #testing
    y_pre = network.predict(test_db,  batch_size=FLAGS.batch_size,
        verbose=1,
        steps=par.val_steps,
        callbacks=None,
        workers=4,
        use_multiprocessing=True,
    )
    y_pre = y_pre/FLAGS.scaling
    
    y_true_ALL = None
    for step, (x,y_true) in enumerate(test_db):
        y_true = y_true.numpy()
        if step == 0:
            y_true_ALL = y_true
        else:
            y_true_ALL = np.concatenate([y_true_ALL, y_true], axis=0)
        if step%1000==0:
            print("Now, adding label to output. Please wait. %d" % step)
    
    
    y_true_ALL = np.concatenate([y_true_ALL[:,0:3]/FLAGS.scaling, y_true_ALL[:,-1].reshape(-1,1)], axis=1)
    print(y_true_ALL)

    output = np.concatenate([y_pre, y_true_ALL], axis=1)
    print(output)
    print(output.shape)
    df = pd.DataFrame(output, columns=['rec_x', 'rec_y', 'rec_z', 'sim_x', 'sim_y', 'sim_z', 'energy'])
    df.to_csv(par.output_file, index=False,header = False )
if __name__ == '__main__':
    main()