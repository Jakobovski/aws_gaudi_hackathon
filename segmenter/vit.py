# This file includes the segmentor model definition. Portions of the file were taken from the
# Habana Gadui ViT sample:
# https://github.com/HabanaAI/Model-References/blob/master/TensorFlow/computer_vision/VisionTransformer/vit_keras/vit.py
import tensorflow as tf
from segmenter import layers, utils

CONFIG_B = {
    "dropout": 0.1,
    "mlp_dim": 3072,
    "num_heads": 12,
    "num_layers": 12,
    "hidden_size": 768,
}

CONFIG_L = {
    "dropout": 0.1,
    "mlp_dim": 4096,
    "num_heads": 16,
    "num_layers": 24,
    "hidden_size": 1024,
}

def build_model(
    image_size: int,
    patch_size: int,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    name: str,
    mlp_dim: int,
    classes: int,
    dropout=0.1,
    num_decoder_layers=2
):
    """Build a ViT model.

    Args:
        image_size: The size of input images.
        patch_size: The size of each patch (must fit evenly in image_size)
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        num_layers: The number of transformer layers to use.
        hidden_size: The number of filters to use
        num_heads: The number of transformer heads
        mlp_dim: The number of dimensions for the MLP output in the transformers.
        dropout: fraction of the units to drop for dense layers.
        num_decoder_layers: The number of transformer layers for decoder. 0 for linear
    """
    assert image_size % patch_size == 0, "image_size must be a multiple of patch_size"
    x = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    y = tf.keras.layers.Conv2D(
        filters=hidden_size,
        kernel_size=patch_size,
        strides=patch_size,
        padding="valid",
        name="embedding",
    )(x)
    y = tf.keras.layers.Reshape((y.shape[1] * y.shape[2], hidden_size))(y)
    y = layers.ClassToken(name="class_token")(y)
    y = layers.AddPositionEmbs(name="Transformer/posembed_input")(y)
    for n in range(num_layers):
        y, _ = layers.TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer/encoderblock_{n}",
        )(y)
    y = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(y)
    y = tf.keras.layers.Lambda(lambda v: v[:, 1:], name="ExtractToken")(y)

    if num_decoder_layers > 0:
        y = layers.EmbedRepresentation(hidden_size, classes)(y)
        for n in range(num_decoder_layers):
            y, _ = layers.TransformerBlock(
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                name=f"Transformer/decoderblock_{n}",
            )(y)
        y = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="Transformer/decoder_norm"
        )(y)
        y = layers.MaskEmbedding(hidden_size,classes)(y)
    else:
        # linear decoder
        y = tf.keras.layers.Dense(classes,name="head")(y)
    y = tf.keras.layers.Lambda(lambda x:tf.image.resize(tf.reshape(tf.nn.softmax(x),[-1,image_size//patch_size,image_size//patch_size,classes]),[image_size,image_size]))(y)
    return tf.keras.models.Model(inputs=x, outputs=y, name=name)


def load_pretrained(size, weights, model):
    """Load model weights for a known configuration."""
    fname = f"ViT-{size}_{weights}.npz"
    import os
    utils.load_weights_numpy(model, f'{os.environ.get("HOME","")}/.keras/weights/{fname}')


def vit_b16(
    image_size: int = 224,
    classes=1000,
    pretrained=True,
    num_decoder_layers=0,
    weights="imagenet21k+imagenet2012"
):
    """Build ViT-B16. All arguments passed to build_model."""
    model = build_model(
        **CONFIG_B,
        name="vit-b16",
        patch_size=16,
        image_size=image_size,
        classes=classes,
        num_decoder_layers=num_decoder_layers
    )
    if pretrained:
        load_pretrained(
            size="B_16", weights=weights, model=model
        )

    return model

def vit_b32(
    image_size: int = 224,
    classes=1000,
    pretrained=True,
    num_decoder_layers=0,
    weights="imagenet21k+imagenet2012"
):
    """Build ViT-B32. All arguments passed to build_model."""
    model = build_model(
        **CONFIG_B,
        name="vit-b32",
        patch_size=32,
        image_size=image_size,
        classes=classes,
        num_decoder_layers=num_decoder_layers
    )
    if pretrained:
        load_pretrained(
            size="B_32", weights=weights, model=model
        )
    return model

def vit_l16(
    image_size: int = 384,
    classes=1000,
    pretrained=True,
    num_decoder_layers=0,
    weights="imagenet21k+imagenet2012"
):
    """Build ViT-L16. All arguments passed to build_model."""
    model = build_model(
        **CONFIG_L,
        patch_size=16,
        name="vit-l16",
        image_size=image_size,
        classes=classes,
        num_decoder_layers=num_decoder_layers
    )
    if pretrained:
        load_pretrained(
            size="L_16", weights=weights, model=model
        )
    return model

def vit_l32(
    image_size: int = 384,
    classes=1000,
    pretrained=True,
    num_decoder_layers=0,
    weights="imagenet21k+imagenet2012"
):
    """Build ViT-L32. All arguments passed to build_model."""
    model = build_model(
        **CONFIG_L,
        patch_size=32,
        name="vit-l32",
        image_size=image_size,
        classes=classes,
        num_decoder_layers=num_decoder_layers
    )
    if pretrained:
        load_pretrained(
            size="L_32", weights=weights, model=model
        )
    return model
