# This file was adopted from Habana Model-References github:
# https://github.com/HabanaAI/Model-References/blob/master/TensorFlow/computer_vision/VisionTransformer/vit_keras/layers.py
# We have appended custom Segmenter decoder layers at the bottom of file

import tensorflow as tf
import tensorflow_addons as tfa


@tf.keras.utils.register_keras_serializable()
class ClassToken(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = tf.keras.layers.Dense(hidden_size, name="query")
        self.key_dense = tf.keras.layers.Dense(hidden_size, name="key")
        self.value_dense = tf.keras.layers.Dense(hidden_size, name="value")
        self.combine_heads = tf.keras.layers.Dense(hidden_size, name="out")

    # pylint: disable=no-self-use
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# pylint: disable=too-many-instance-attributes
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0",
                ),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False)
                )
                if hasattr(tf.keras.activations, "gelu")
                else tf.keras.layers.Lambda(
                    lambda x: tfa.activations.gelu(x, approximate=False)
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Segmenter decoder layers
class EmbedRepresentation(tf.keras.layers.Layer):
    def __init__(self, projection_dim, num_classes):
        super(EmbedRepresentation, self).__init__()
        self.num_classes = num_classes
        self.projection_dim = projection_dim
        self.class_emb = self.add_weight("class_emb", shape=(1, num_classes, projection_dim))
        self.projection = tf.keras.layers.Dense(units=projection_dim)


    def call(self, x):
        # remove CLS/DIST tokens for decoding
        x = self.projection(x)
        batch_size = tf.shape(x)[0]
        class_emb = tf.broadcast_to(self.class_emb, [batch_size, self.num_classes, self.projection_dim])
        x = tf.concat([x,class_emb], axis=1)
        return x

    def get_config(self):
        config = super(EmbedRepresentation, self).get_config()
        config.update({'num_classes': self.num_classes})
        config.update({'projection_dim': self.projection_dim})
        return config

class ExampleRandomNormal(tf.keras.initializers.Initializer):

  def __init__(self, mean, stddev, scale):
    self.mean = mean
    self.stddev = stddev
    self.scale = scale

  def __call__(self, shape, dtype=None, **kwargs):
    return self.scale*tf.random.normal(
        shape, mean=self.mean, stddev=self.stddev, dtype=dtype)

  def get_config(self):  # To support serialization
    return {"mean": self.mean, "stddev": self.stddev, "scale":self.scale}

class MaskEmbedding(tf.keras.layers.Layer):
    def __init__(self, projection_dim, num_classes):
        super(MaskEmbedding, self).__init__()
        self.scale = projection_dim ** -0.5
        self.num_classes = num_classes
        self.projection_dim = projection_dim
        self.proj_patch = self.add_weight('proj_patch', [projection_dim,projection_dim], initializer = ExampleRandomNormal(mean=0., stddev = 1.,scale = self.scale))
        self.proj_classes = self.add_weight('proj_classes', [projection_dim,projection_dim], initializer = ExampleRandomNormal(mean=0., stddev = 1.,scale = self.scale))
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.mask_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        patches, cls_seg_feat = tf.split(x,[-1,self.num_classes],1)
        patches = tf.linalg.matmul(patches, self.proj_patch)
        cls_seg_feat = tf.linalg.matmul(cls_seg_feat, self.proj_classes)
        patches = tf.math.l2_normalize(patches,axis=-1)
        cls_seg_feat = tf.math.l2_normalize(cls_seg_feat, axis=-1)
        masks = tf.linalg.matmul(patches, cls_seg_feat, transpose_b=True)
        masks = self.mask_norm(masks)
        return masks

    def get_config(self):
        config = super(MaskEmbedding, self).get_config()
        config.update({'num_classes': self.num_classes})
        config.update({'projection_dim': self.projection_dim})
        return config

# specialized loss and metric classes that account for input label format,
# specifcally the ignore label ('0')
# avoid the use of ops with dynamic shape outputs such as tf.boolean_mask
# that don't perform optimally on HPU
class SparseCategoricalCrossentropyEx(tf.keras.losses.SparseCategoricalCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(reduction=tf.losses.Reduction.NONE,**kwargs)

    def call(self,y_true, y_pred):
        mask = tf.not_equal(y_true,0)
        y_true = tf.where(mask,y_true-1, y_true)
        cost = super(SparseCategoricalCrossentropyEx,self).call(y_true,y_pred)
        cost = cost * tf.cast(mask, cost.dtype)
        cost =  tf.math.divide_no_nan(tf.reduce_sum(cost),tf.math.count_nonzero(mask,dtype=cost.dtype))
        return cost

class SparseCategoricalAccuracyEx(tf.keras.metrics.SparseCategoricalAccuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        assert not sample_weight
        mask = tf.not_equal(y_true,0)
        y_true = tf.where(mask,y_true-1, y_true)
        return super(SparseCategoricalAccuracyEx,self).update_state(y_true,y_pred,mask)

class SparseCategoricalMeanIoU(tf.keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        assert not sample_weight
        mask = tf.not_equal(y_true,0)
        y_true = tf.where(mask,y_true-1, y_true)
        y_pred = tf.math.argmax(y_pred, axis=-1,output_type=tf.dtypes.int32)
        return super(SparseCategoricalMeanIoU,self).update_state(y_true,y_pred,mask)
