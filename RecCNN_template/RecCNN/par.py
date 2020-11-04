#coding:utf-8
'''
hyperparameters
'''

import tensorflow as tf

FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_integer('epochs', 10, """Number of epoch to train.""")
tf.compat.v1.flags.DEFINE_integer('batch_size', 64, """Number of batches to run.""")
tf.compat.v1.flags.DEFINE_boolean('is_training', True, """Is training or not.""")
tf.compat.v1.flags.DEFINE_float('scaling', 0.01, """The linear scaling parameter of unit [mm]""")
tf.compat.v1.flags.DEFINE_float('time_scaling', 200, """The linear scaling parameter of unit for first hit time [ns]""")
tf.compat.v1.flags.DEFINE_integer('train_mindocid', 0, """mindocid for trainning dataset""")
tf.compat.v1.flags.DEFINE_integer('train_maxdocid', 90, """maxdocid for trainning dataset""")
tf.compat.v1.flags.DEFINE_integer('val_mindocid', 90, """mindocid for valid dataset""")
tf.compat.v1.flags.DEFINE_integer('val_maxdocid', 100, """maxdocid for valid dataset""")
tf.compat.v1.flags.DEFINE_integer('nevt_file', 500, """number of events per file""")
tf.compat.v1.flags.DEFINE_string('particle', 'eplus_ekin5M', '''The particle type''')
tf.compat.v1.flags.DEFINE_string('modelname', 'modelname', """The modelname.""")

tf.compat.v1.flags.DEFINE_integer('mindocID', 0, """mindocID for valid dataset.""")
tf.compat.v1.flags.DEFINE_integer('maxdocID', 100, """maxdocID for valid dataset.""")
tf.compat.v1.flags.DEFINE_integer('modelID', 1, """model id for rec""")

'------------------train---------------'
path_to_tfr = '1'
path_to_model = './result/'
path_to_result = '2'
epoch_size = 0
steps_per_epoch = 0
total_steps = 0
path_to_tfboard = './tfboard/'

val_step = 0
output_file = '1'

def init_train_par():
    global path_to_tfr
    global path_to_model
    global path_to_result
    global epoch_size
    global total_steps
    global steps_per_epoch
    global path_to_tfboard
    path_to_tfr = './data/' + FLAGS.particle + '/'
    path_to_result = path_to_model + FLAGS.modelname + '/' 
    path_to_tfboard = path_to_tfboard + FLAGS.modelname + '/' 
    epoch_size = (FLAGS.train_maxdocid-FLAGS.train_mindocid) * FLAGS.nevt_file
    steps_per_epoch = epoch_size // FLAGS.batch_size
    total_steps = steps_per_epoch * FLAGS.epochs 
   
    print("**************** PARAMETERS *****************")
    print("Modelname = ", FLAGS.modelname)
    print("Total number of events = ", epoch_size)
    print("unit_scale = ", FLAGS.scaling) 
    print("epoch_num = %d, batch size = %d, in total %d steps" % (FLAGS.epochs, FLAGS.batch_size, total_steps))
    print("data dir %s" % path_to_tfr)
    print("**************** PARAMETERS *****************")

    
'------------------test---------------'
def init_test_par():
    FLAGS.nevt_file = 100
    FLAGS.batch_size = 64
    FLAGS.is_training = False
    FLAGS.particle = 'eplus'
    global path_to_model
    global path_to_tfr
    global path_to_result
    global epoch_size
    global val_steps
    global output_file
    
    path_to_model = '../train/result/' + str(FLAGS.modelname) + '/' + str(FLAGS.modelname) + '_' + str(FLAGS.modelID)
    path_to_tfr = './data/' + FLAGS.particle + '/'
    path_to_result = './result/' + FLAGS.modelname + '/'
    epoch_size = (FLAGS.maxdocID-FLAGS.mindocID) * FLAGS.nevt_file
    val_steps = epoch_size / FLAGS.batch_size
    output_name = 'recvtx_' + str(FLAGS.modelID) + '.csv'
    output_file = path_to_result + output_name

    print("**************** PARAMETERS *****************")
    print("Modelname = ", FLAGS.modelname)
    print("is training = ", FLAGS.is_training)
    print("unit_scale = ", FLAGS.scaling) 
    print("test dir %s" % path_to_tfr)
    print("test doc from %d to %d" %(FLAGS.mindocID,FLAGS.maxdocID))
    print("test model ID = %d" % FLAGS.modelID)
    print("test batch size = %d " % FLAGS.batch_size)
    print("total events = %d" % (epoch_size))
    print("num of step to run = %d" %  val_steps)
    print("out file is %s " % output_file)
    print("**************** PARAMETERS *****************")

