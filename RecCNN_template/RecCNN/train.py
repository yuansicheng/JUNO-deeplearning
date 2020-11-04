
#coding:utf-8
import tensorflow as tf

from .par import FLAGS

def mean_square_loss(y_true, y_pred):
    '''
    loss function
    '''
    y_tr = y_true[:, 0:3]

    cross_entropy = tf.reduce_mean(tf.pow((y_tr - y_pred), 2), axis=1) 
    #cross_entropy = tf.pow(cross_entropy, 0.5)
    cross_entropy = tf.reduce_mean(cross_entropy)

    return cross_entropy

def stage_weight_loss(y_true, y_pred):
    '''
    loss of multiple output for ResNet
    '''

    y_tr = y_true[:, 0:3]

    loss = tf.reduce_mean(tf.pow(y_pred[:, 0:3] - y_tr, 2), axis=1) 
    loss = tf.reduce_mean(loss)

    loss3 = tf.reduce_mean(tf.pow(y_pred[:, 6:9] - y_tr, 2), axis=1) 
    loss3 = tf.reduce_mean(loss3)

    loss4 = tf.reduce_mean(tf.pow(y_pred[:, 9:12] - y_tr, 2), axis=1) 
    loss4 = tf.reduce_mean(loss4)

    loss = loss + 0.2 * loss3 + 0.3 * loss4

    return loss

def res(y_true, y_pred):
    '''
    Get result's accuracy for validation
    '''
    # y_ture: batch_size * 12
    values = tf.reduce_sum(tf.pow(y_pred[:, 0:3] - y_true[:, 0:3], 2), axis=1)
    values = tf.sqrt(values)
    values = tf.reduce_mean(tf.pow(values, 0.5)) / FLAGS.scaling

    return values