import tensorflow as tf
import tflearn

def _conv_2d(layer,num_filters,filter_size,strides,activation='relu',weight_decay=1e-5,regularizer='L2',bn=True):
	layer = tflearn.layers.conv.conv_2d(layer,num_filters,filter_size,strides=strides,activation=activation,weight_decay=weight_decay,regularizer=regularizer)
	if bn:
		layer = tflearn.layers.normalization.batch_normalization(layer)
	return layer


def _conv_2d_transpose(layer,num_filters,filter_size,output_size,strides,activation='relu',weight_decay=1e-5,regularizer='L2',bn=True):
	layer=tflearn.layers.conv.conv_2d_transpose(layer,num_filters,filter_size,output_size,strides=strides,activation=activation,weight_decay=weight_decay,regularizer=regularizer)
	if bn:
		layer=tflearn.layers.normalization.batch_normalization(layer)
	return layer


def _fc(layer,num_neurons,activation='relu',weight_decay=1e-3,regularizer='L2'):
	layer = tflearn.layers.core.fully_connected(layer,num_neurons,activation=activation,weight_decay=weight_decay,regularizer=regularizer)
	return layer
