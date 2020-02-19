import tensorflow as tf

#  define common parameter
tf.app.flags.DEFINE_bool('isTraining','True','define if is train or test')
tf.app.flags.DEFINE_integer('num_classes',2,'The number of classes')
tf.app.flags.DEFINE_integer('imgSize',512,'image size')
tf.app.flags.DEFINE_integer('img_channel',3,'image channel')
tf.app.flags.DEFINE_float('noamlization_factor',1.,'')
tf.app.flags.DEFINE_string('dataSet',r'tfRecords','')
tf.app.flags.DEFINE_string('logdir',r'log\log','')
tf.app.flags.DEFINE_string('model_path',r'model\model','')
#  define training parameter
tf.app.flags.DEFINE_integer('iters',500000,'')
tf.app.flags.DEFINE_float('learning_rate',0.0001,'learning rate')
tf.app.flags.DEFINE_integer('batch_size',1,'')
tf.app.flags.DEFINE_integer('test_batch_size','5','')
tf.app.flags.DEFINE_string('testDataSet',r'tfRecords','')
tf.app.flags.DEFINE_string('output_path',r'','')
tf.app.flags.DEFINE_string('data_format','*.tiff','')
tf.app.flags.DEFINE_string('xml_path','','')


FLAGS=tf.app.flags.FLAGS

