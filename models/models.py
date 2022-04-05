# This file consists of model definitions that we'll be used for hyperparameter search and tuning.

import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPool2D, Flatten, AvgPool2D, Dropout


# EfficientNetB3
class XrayModelEfficient(tf.keras.Model):
    def __init__(self, dropout_rate):
        super(XrayModelEfficient, self).__init__()

        self.base_model = tf.keras.applications.EfficientNetB3(input_shape=(150, 150, 3),
                                                               include_top=False,
                                                               weights='imagenet')
        for layer in self.base_model.layers:
            layer.trainable = False
        for layer in self.base_model.layers[-10:]:
            layer.trainable = True

        self.pool = AvgPool2D()
        self.flatten = Flatten()
        self.dropout = Dropout(dropout_rate)
        self.classifier = Dense(3, activation='softmax')
    
    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        outputs = self.classifier(x)
        return outputs


# InceptionV3
class XrayModelInception(tf.keras.Model):
    def __init__(self, dropout_rate):
        super(XrayModelInception, self).__init__()

        self.base_model = tf.keras.applications.InceptionV3(input_shape=(150, 150, 3),
                                                            include_top=False,
                                                            weights='imagenet')
        for layer in self.base_model.layers:
            layer.trainable = False
        for layer in self.base_model.layers[-10:]:
            layer.trainable = True

        self.pool = AvgPool2D()
        self.flatten = Flatten()
        self.dropout = Dropout(dropout_rate)
        self.classifier = Dense(3, activation='softmax')
    
    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        outputs = self.classifier(x)
        return outputs


# MobileNetV2
class XrayModelMobilenet(tf.keras.Model):
    def __init__(self, dropout_rate):
        super(XrayModelMobilenet, self).__init__()

        self.base_model = tf.keras.applications.MobileNetV2(input_shape=(150, 150, 3),
                                                            include_top=False,
                                                            weights='imagenet')
        for layer in self.base_model.layers:
            layer.trainable = False
        for layer in self.base_model.layers[-10:]:
            layer.trainable = True

        self.pool = MaxPool2D()
        self.flatten = Flatten()
        self.dropout = Dropout(dropout_rate)
        self.classifier = Dense(3, activation='softmax')
    
    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        outputs = self.classifier(x)
        return outputs