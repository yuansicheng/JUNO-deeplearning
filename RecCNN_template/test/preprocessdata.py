# the shape of 2d projection is (x,y).shape = (230,124)
# make 2d proj for hits_data, and transfer to Tensorflow data format, i.e. tfrecords

import numpy as np
import tensorflow as tf
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

FLAGS = tf.compat.v1.flags.FLAGS
tf.compat.v1.flags.DEFINE_integer('mindocID', 0, '''The min docid in hits_data files''')
tf.compat.v1.flags.DEFINE_integer('maxdocID', 1100, '''The max docid in hits_data files''')
tf.compat.v1.flags.DEFINE_string('particle', 'eplus', '''The particle type''')

# output dir
#path_tfr = './data/tfr/'
path_tfr = './data/' + FLAGS.particle + '/'
if not os.path.exists(path_tfr):
    os.makedirs(path_tfr)

# map file
pmt_pos  = np.genfromtxt('../id2pos/id2pos_interval.txt', delimiter=',', dtype=int)

# hits and sim files
path_pmt = './data/hits_data'
path_sim = './data/sim_data'
if not os.path.exists(path_pmt):
    print('input hits_data not exist, make sure the following data set is the desired one')
    
if not os.path.exists(path_sim):
    print('input sim_data not exist, make sure the following data set is the desired one')
    


def find_pos(pmtid):
    x = pmt_pos[pmtid, 1]
    y = pmt_pos[pmtid, 2]

    return x, y


def input_data(docid):
    print('loading file %d' % docid)

    tl0 = time.time()  # >>>> -----
    eve_dict = {}
    label_dict = {}

    dir_pmt = path_pmt + '/hits_' + str(docid) + '.txt'
    dir_sim = path_sim + '/sim_' + str(docid) + '.txt'

    doc_pmt = pd.read_csv(dir_pmt, sep=" ", header=None).values
    doc_sim = pd.read_csv(dir_sim, sep=" ", header=None).values

    tl1 = time.time() - tl0  # ----- <<<<
    print("loading %.2f sec/file" % (tl1))

    td0 = time.time()

    mark_start = 0
    mark_end = 0

    pmt_row_num = doc_pmt.shape[0]
    sim_row_num = doc_sim.shape[0]
    print('shape of doc_pmt', doc_pmt.shape)
    print('shape of doc_sim', doc_sim.shape)

    valid_eve_num = 0

    for eve_num in range(sim_row_num):
        if eve_num % 100 == 99:
            print('generating dict. event %d/%d' % (eve_num, sim_row_num))

        mark_start = mark_end

        while doc_pmt[mark_end, 0] == doc_pmt[mark_start, 0]: #make mark_end point to next eve_num
            mark_end += 1
            if mark_end == pmt_row_num:
                break

        if eve_num == doc_pmt[mark_start, 0]:  # if this eve_num number has hits
            if not doc_sim[eve_num, 4] == 0:   # remove events with 0 edep (expecially for gamma)
                eve_str = 'eve' + str(valid_eve_num)
                label_str = 'label' + str(valid_eve_num)

                label_dict[label_str] = doc_sim[eve_num, :]
                if mark_end == pmt_row_num:
                    eve_dict[eve_str] = doc_pmt[mark_start:, :]
                    break   # incase 99 in sim while only 98 in hits
                else:
                    eve_dict[eve_str] = doc_pmt[mark_start:mark_end, :]
            
                valid_eve_num += 1

        else:
            mark_end = mark_start  # if the doc_pmt[mark_start, 0] is bigger, wait for the next eve_num

    td1 = time.time() - td0
    print("dicting %.2f sec/file" % (td1))
    print("lenth of eve_dict = ", len(eve_dict)) 
    print("lenth of label_dict = ", len(label_dict)) 

    return eve_dict, label_dict


def project_output(eve_dict, label_dict, docid):
    eve_num = len(eve_dict)

    t_project = 0
    t_write = 0

    for i in range(eve_num):
        tp0 = time.time()  # >>> -----

        one_eve = eve_dict['eve' + str(i)]  # select out one event for iteration
        x_grid = np.zeros((230, 124, 2), dtype=np.float32)

        for j in range(one_eve.shape[0]): # all hits in one event
            pmtid = int(one_eve[j, 1])
            x, y = find_pos(pmtid)
            x_grid[x, y, 0] = one_eve[j, 2]  # store the hit time
            x_grid[x, y, 1] = one_eve[j, 3]  # store the nPE
            
        one_label = label_dict['label' + str(i)]
        y_grid = np.array(one_label[1:-1], dtype=np.float32)  # (eve,x,y,z,energy) in sim_x.txt
        energy = np.float32(one_label[-1])

        tp1 = time.time() - tp0
        t_project += tp1  # ----- <<<<

        tw0 = time.time()  # >>> -----

        features = {}

        features['data'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_grid.tostring()]))
        features['label'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[y_grid.tostring()]))
        features['energy'] = tf.train.Feature(float_list=tf.train.FloatList(value=[energy]))

        tf_features = tf.train.Features(feature=features)
        tf_example = tf.train.Example(features=tf_features)
        tf_serialized = tf_example.SerializeToString()

        tfc_writer.write(tf_serialized)

        tw1 = time.time() - tw0
        t_write += tw1  # ----- <<<<


    print('project %.2f sec/file, write %.2f sec/file\n' % (t_project, t_write))


# ---------- Start Preprocess Data ----------
for i in range(FLAGS.mindocID, FLAGS.maxdocID):
    tfc_name = path_tfr + FLAGS.particle + '_' + str(i) + '.tfrecords'
    tfc_writer  = tf.io.TFRecordWriter(tfc_name)

    eve_dict_i, label_dict_i = input_data(i)
    project_output(eve_dict_i, label_dict_i, i)

    tfc_writer.close()
