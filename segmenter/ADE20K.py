import os
from functools import partial

import tensorflow as tf

image_size = 512
num_classes = 150
train_samples = 20200
test_samples = 2000

def _parse_fn(example_serialized, is_training):
    feature_map = {
        'image': tf.io.FixedLenFeature([], dtype=tf.string,
                                               default_value=''),
        'labels': tf.io.FixedLenFeature([], dtype=tf.string,
                                                   default_value=''),
    }
    parsed = tf.io.parse_single_example(
        serialized=example_serialized, features=feature_map)
    image = tf.image.decode_jpeg(parsed['image'])
    labels = tf.image.decode_png(parsed['labels'])
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    sh = tf.cast(tf.shape(image)[:2],tf.float32)
    h_f = sh[0]
    w_f = sh[1]
    scale = tf.reduce_max([image_size/h_f,image_size/w_f])
    if is_training:
        min_scale = tf.reduce_max([scale,0.5])
        max_scale = tf.reduce_max([scale,2.])
        scale = tf.random.uniform((),min_scale,max_scale)
        source_size = tf.cast(tf.math.round(image_size / scale),tf.int32)
        crop_base_h = tf.random.uniform((), maxval=h-source_size+1,dtype=tf.int32)
        crop_base_w = tf.random.uniform((), maxval=w-source_size+1,dtype=tf.int32)
    else:
        #central crop
        source_size = tf.minimum(h_f,w_f)
        crop_base_h = tf.cast(tf.math.round((h_f - source_size) / 2),tf.int32)
        crop_base_w = tf.cast(tf.math.round((w_f - source_size) / 2),tf.int32)
        source_size = tf.cast(source_size,tf.int32)

    image = tf.image.resize(image[crop_base_h:crop_base_h+source_size+1,crop_base_w:crop_base_w+source_size+1], [image_size, image_size],
                                         method=tf.image.ResizeMethod.BILINEAR)
    labels = tf.image.resize(labels[crop_base_h:crop_base_h + source_size + 1, crop_base_w:crop_base_w + source_size + 1],
                            [image_size, image_size],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if is_training:
        flip = tf.random.uniform((), maxval=1.) < 0.5
        image = tf.cond(flip, lambda: tf.reverse(image, [1]), lambda: image)
        labels = tf.cond(flip, lambda: tf.reverse(labels, [1]), lambda: labels)
        image = tf.cast(image, tf.float32)
        brightness = tf.random.uniform((), maxval=1.) < 0.5
        b = tf.where(brightness,tf.random.uniform((), -32.,32.),0.)
        image = tf.clip_by_value(image+b,0.,255.)
        contrast = tf.random.uniform((), maxval=1.) < 0.5
        c = tf.where(contrast, tf.random.uniform((), 0.5, 1.5), 1.)
        image = tf.clip_by_value(image*c, 0., 255.)
    else:
        image = tf.cast(image, tf.float32)

    image = (image/255.-0.5)*2.
    labels = tf.reshape(labels,[image_size,image_size])
    image = tf.reshape(image,[image_size,image_size,3])
    return (image, labels)

def get_dataset(tfrecords_dir, batch_size, is_training):
    """Read TFRecords files and turn them into a TFRecordDataset.

    Args:
        tfrecords_dir: dataset directory
        subset: pattern to detect subset in dataset directory
        batch_size: Global batch size
        is_training (bool): use True if dataset will be used for training

    Returns:
        TFRecordDataset: Dataset.
    """
    ds = tf.data.Dataset.list_files(os.path.join(tfrecords_dir, 'train' if is_training else 'test','*.tfrecord'),shuffle=is_training)
    ds = tf.data.TFRecordDataset(ds)
    ds = ds.repeat()

    if is_training:
        ds = ds.shuffle(buffer_size=1000)

    parser = partial(_parse_fn, is_training=is_training)
    ds = ds.map(map_func=parser,
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size=batch_size, drop_remainder=True)
    return ds
