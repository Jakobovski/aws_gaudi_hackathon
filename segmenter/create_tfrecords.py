"""Script to convert ADE20K to tfrecords
python create_tfrecords.py \
  --input_path="./data" \
  --output_path="./tf_records"
"""
import os
from absl import app
from absl import flags
import tensorflow as tf
import cv2
import numpy as np
flags.DEFINE_string(
    'input_path', "./ADEChallengeData2016", 'location of raw data.')
flags.DEFINE_string(
    'output_path', "./tf_records", 'output location.')
flags.DEFINE_string(
    'num_records', None, 'Number of records to parse.')

FLAGS = flags.FLAGS

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def to_tfrecord_file(in_data_dir,inds,outfile,test):
    tag = 'train'
    subdir = 'training'
    if test:
        tag = 'val'
        subdir = 'validation'
    with tf.io.TFRecordWriter(outfile) as writer:
        for i in inds:
            print(i)
            image_path = os.path.join(in_data_dir,'images',subdir,f'ADE_{tag}_{i:08}.jpg')
            annot_path = os.path.join(in_data_dir,'annotations',subdir,f'ADE_{tag}_{i:08}.png')
            with open(image_path,'rb') as image, open(annot_path,'rb') as labels:
                image_buf = image.read()
                img = cv2.imdecode(np.fromstring(image_buf, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if len(list(img.shape))< 3 or img.shape[2] != 3:
                    assert 0
                labels_buf = labels.read()
                example = tf.train.Example(features=tf.train.Features(feature={
                    'labels': _bytes_feature(labels_buf),
                    'image': _bytes_feature(image_buf)}))
            writer.write(example.SerializeToString())
        writer.close()

def main(args):
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    in_data_dir = FLAGS.input_path
    out_data_dir = FLAGS.output_path
    num_train = 20210
    num_val = 2000
    import random
    train = list(range(1,num_train+1))
    # remove grayscale images
    train = [i for i in train if i not in [1701,3020,8455,13508]]
    random.shuffle(train)
    train_shards = 8
    records_per_shard = len(train) // 8
    print(f'train size {records_per_shard*8}')
    val = range(1,num_val+1)

    # create train tf records
    for i in range(train_shards):
        outfile = os.path.join(os.path.join(out_data_dir, 'train', f'{i}.tfrecord'))
        to_tfrecord_file(in_data_dir, train[i*records_per_shard:(i+1)*records_per_shard], outfile, test=False)
    # create test tf records
    outfile = os.path.join(os.path.join(out_data_dir,'test','0.tfrecord'))
    to_tfrecord_file(in_data_dir,val,outfile,test=True)

if __name__ == '__main__':
    app.run(main)
