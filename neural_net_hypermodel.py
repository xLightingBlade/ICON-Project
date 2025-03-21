from keras_tuner import HyperModel
import tensorflow as tf
import numpy as np

class MyHypermodel(HyperModel):
    def __init__(self, inputs):
        self.inputs = inputs

    def build(self, hp):
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(self.inputs))
        dense_units = hp.Int('units', min_value=32, max_value=256, step=32)
        model = tf.keras.models.Sequential([
            tf.keras.Input(shape=(30,)),
            normalizer,
            tf.keras.layers.Dense(units=dense_units, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(units=dense_units, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(units=dense_units, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
            tf.keras.layers.Dense(units=1, activation='sigmoid'),]
        )

        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001),
                      loss="binary_crossentropy",
                      metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
        model.summary()
        return model