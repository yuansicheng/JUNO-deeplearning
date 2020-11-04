#coding:utf-8
#author: Ziyuan Li, Zhu jiang, Yue Huang, Zhen Qian
import tensorflow as tf
import os

from . import par
FLAGS = par.FLAGS

#features: A dict mapping feature keys to FixedLenFeature or VarLenFeature values.
features = {
    'data': tf.io.FixedLenFeature([], dtype=tf.string),
    'label': tf.io.FixedLenFeature([], dtype=tf.string),
    'energy': tf.io.FixedLenFeature([], tf.float32)
}

def parse_function(example_proto):
    """
    When calling a dataset's map(), it will apply this function to each element of the dataset,
    and returns a new dataset containing the transformed elements, in the same order as they appeared in the input.
    """
    parsed_example = tf.io.parse_single_example(example_proto, features)

    x_tr = tf.io.decode_raw(parsed_example['data'], tf.float32)
    y_tr = tf.io.decode_raw(parsed_example['label'], tf.float32) * FLAGS.scaling # sacling 
    y_tr_en = parsed_example['energy']

    x_tr = tf.reshape(x_tr, [230, 124, 2])
    x_tr_hittime = tf.slice(x_tr, [0, 0, 0], [230, 124, 1])
    x_tr_npe = tf.slice(x_tr, [0, 0, 1], [230, 124, 1])

    has_hit = tf.cast(tf.cast(x_tr_npe, dtype=bool), dtype=tf.float32)
    x_tr_hittime = has_hit * x_tr_hittime
    x_tr_hittime = x_tr_hittime / FLAGS.time_scaling # normalize for 200 ns

    y_tr = tf.reshape(y_tr, [3])
    y_tr_en = tf.reshape(y_tr_en, [1])

    x_tr = tf.concat([x_tr_hittime, x_tr_npe], axis=-1)
    y_tr = tf.concat([y_tr, y_tr_en], -1)

    return x_tr, y_tr


def get_dataset(path_to_tfr = par.path_to_tfr, batchsz = FLAGS.batch_size, testdataset = False):
    '''
    read *.tfrecord，and return 2 datasets
    '''
    filename = [os.path.join(path_to_tfr, file) for file in os.listdir(path_to_tfr)]

    
    if not filename == None:
        #training dataset
        if not testdataset:
            #training dataset
            train_name = filename[FLAGS.train_mindocid: FLAGS.train_maxdocid]
            train_set = tf.data.Dataset.from_tensor_slices(train_name)
            train_set = train_set.interleave(tf.data.TFRecordDataset, cycle_length=50, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            train_set = train_set.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batchsz).shuffle(batchsz).repeat()
            train_set = train_set.prefetch(2)

            #validating dataset
            test_name = filename[FLAGS.val_mindocid: FLAGS.val_maxdocid]
            test_set = tf.data.Dataset.from_tensor_slices(test_name)
            test_set = test_set.interleave(tf.data.TFRecordDataset, cycle_length=50, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            test_set = test_set.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batchsz).repeat().prefetch(2)
        # testing dataset
        else:
            test_name = filename[FLAGS.mindocID: FLAGS.maxdocID]
            test_set = tf.data.Dataset.from_tensor_slices(test_name)
            test_set = test_set.interleave(tf.data.TFRecordDataset, cycle_length=50, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            test_set = test_set.map(parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batchsz).prefetch(2)
            train_set = None
    else:
        train_set = None
        test_set = None
        print("ERROR: No files in {}!\nPlease Check!".format(path_to_tfr))
    

    return train_set, test_set

