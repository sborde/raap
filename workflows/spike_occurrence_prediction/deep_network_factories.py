"""Classes and function for the contrastive CNN network."""

from pathlib import Path
import keras
from keras import layers as kl
import numpy as np
import pandas as pd
import tensorflow as tf


def create_dnn_network(
    input_shape: tuple,
    embedding_dimension: int,
    layers: list[dict]
):
    """Creates a simple DNN model for generating embeddings from input data."""
    inputs = keras.Input(shape=input_shape)

    x = inputs
    for layer_param in layers:
        x = kl.Dense(
            layer_param['units'],
            activation=layer_param['activation']
        )(x)
        if 'dropout_rate' in layer_param:
            x = kl.Dropout(layer_param['dropout_rate'])(x)

    outputs = kl.Dense(embedding_dimension)(x)
    return keras.Model(inputs, outputs)

def create_cnn_network(layers: dict):
    """Creates a simple CNN model for generating embeddings from input data."""
    inputs = keras.Input(shape=layers['input']['shape'])

    x = inputs
    for layer_param in layers['conv']:
        if layer_param['type'] == 'Conv1D':
            x = kl.Conv1D(
                layer_param['filters'],
                layer_param['kernel_size'],
                padding='same',
                activation=layer_param['activation']
            )(x)
        if layer_param['type'] == 'MaxPooling1D':
            x = kl.MaxPooling1D(layer_param['pool_size'])(x)

    if layers['flatten']['type'] == 'Flatten':
        x = kl.Flatten()(x)
    elif layers['flatten']['type'] == 'GlobalAveragePooling1D':
        x = kl.GlobalAveragePooling1D()(x)

    for layer_param in layers['fcn']:
        if layer_param['type'] == 'Dense':
            x = kl.Dense(
                layer_param['units'],
                activation=layer_param['activation']
            )(x)
            continue
        if layer_param['type'] == 'Dropout':
            x = kl.Dropout(layer_param['rate'])(x)

    outputs = kl.Dense(
        layers['output']['units'],
        activation=layers['output']['activation']
    )(x)

    return keras.Model(inputs, outputs)

def create_optimizer(name, learning_rate):
    """Creates an optimizer based on the specified type and learning rate."""
    if name == 'adam':
        return keras.optimizers.Adam(learning_rate=learning_rate)

    raise ValueError(f"Unsupported optimizer: {name}")


def create_dnn_model(
    input_shape: tuple,
    output_dimension: int,
    layer_params: list[dict],
    learning_rate: float = 0.001
) -> keras.Model:
    """Creates and compiles the embedding model."""

    model = create_dnn_network(input_shape, output_dimension, layer_params)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss)
    return model

def create_cnn_model(
    layer_params: dict,
    learning_rate: float = 0.001
) -> keras.Model:
    """Creates and compiles the embedding model."""

    model = create_cnn_network(layer_params)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optimizer, loss=loss)
    return model
