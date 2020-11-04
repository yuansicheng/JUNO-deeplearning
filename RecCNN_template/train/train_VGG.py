#coding:utf-8
#author: Ziyuan Li, Zhu jiang, Yue Huang, Zhen Qian

import os, sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # log level
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # select a GPU card

sys.path.append("..")
from RecCNN import par
from RecCNN.data import get_dataset
from RecCNN.VGG16 import VGG16
from RecCNN.ResNet import ResNet
from RecCNN.train import stage_weight_loss, res, mean_square_loss

FLAGS = par.FLAGS
FLAGS.modelname = os.path.splitext(os.path.basename(__file__))[0]
par.init_train_par()

#mkdir dir for models' results
if not os.path.exists(par.path_to_result):
    os.makedirs(par.path_to_result)
    os.makedirs(par.path_to_tfboard)

#tfrecord to 2 dataset (train_db(training)，val_db(validating))
train_db, val_db = get_dataset(par.path_to_tfr, FLAGS.batch_size)

#set callback function, every epoch save the model
callbacks_lists = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath= par.path_to_result+FLAGS.modelname+"_{epoch}.hdf5", # "weights-{epoch:02d}.hdf5",
        monitor='val_loss',
        save_weights_only=False,
        save_best_only=True,
        verbose=1),

    tf.keras.callbacks.TensorBoard(log_dir=par.path_to_tfboard)
]

def main():
    #select a model
    network = VGG16()
    
    #select a optimizer
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    #build the model
    #VGG16
    network.compile(optimizer=adam, loss=mean_square_loss, metrics=[res])
    #ResNet
    #network.compile(optimizer=adam, loss=stage_weight_loss, metrics=[res])
    
    #training
    network.fit(train_db, epochs=FLAGS.epochs, steps_per_epoch = par.steps_per_epoch,
                validation_data=val_db, validation_steps=1000, validation_freq=1,
                callbacks=callbacks_lists)
    network.summary()

    #evaluating
    network.evaluate(val_db, steps=3)
    

if __name__ == '__main__':
    main()
    