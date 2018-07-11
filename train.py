'''
Code for training a neural network that takes in an image of an object and viewing angle theta as input, 
and generates an image of that object from the given theta.
Run as:
	python train.py --exp <experiment name> --gpu <gpu id> --data_dir <data directory>
'''

import os
import sys
import tensorflow as tf
import numpy as np
import random
import re
import time
import argparse
from os.path import join

from dataloader import *
from net import multiview_net

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, required=True, help='Name of Experiment Prefixed with index')
parser.add_argument('--gpu', type=str, required=True, help='GPU to use')
parser.add_argument('--data_dir', type=str, required=True, help='Data directory')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training')
parser.add_argument('--val_batch_size', type=int, default=32, help='Batch Size during training')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
parser.add_argument('--max_epoch', type=int, default=20, help='Maximum number of epochs to train for')

FLAGS = parser.parse_args()

print '-='*50
print FLAGS
print '-='*50

# seed for reproducibility
random.seed(1024)
np.random.seed(1024)
tf.set_random_seed(1024)

# Set gpu id
os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

# Training parameters
BATCH_SIZE = FLAGS.batch_size
VAL_BATCH_SIZE = FLAGS.val_batch_size
HEIGHT = 128
WIDTH = 128
NUM_VIEWS = 12
NUM_GT = 8
TF_SUMMARY_MAX_OUT = 12

DATA_DIR = FLAGS.data_dir
EXP_DIR = join('expts',FLAGS.exp)

def get_epoch_loss(val_models, val_pair_indices):
	''' Validation loss calculation 
	Args:
		val_models --> val model paths list
		val_pair_indices --> indices at which we want to query the models list
	Returns:
		val_loss --> validation loss (L2 between GT and predicted images)
		summ --> val summary writer
	'''
	batches = len(val_pair_indices)/VAL_BATCH_SIZE
	val_loss = 0.
	for b in xrange(batches):
		batch_ip, batch_theta, batch_gt = fetch_batch(val_models, val_pair_indices, b, VAL_BATCH_SIZE)
		L, summ = sess.run([loss, merged_summ], feed_dict={img_inp:batch_ip, theta:batch_theta, img_gt:batch_gt})
		val_loss += L/batches
	
	return val_loss, summ


if __name__=='__main__':

	# Create a folder for the experiment and copy the training file
	create_folder(EXP_DIR)
	os.system('cp train.py %s'%EXP_DIR)
	with open(join(EXP_DIR, 'settings.txt'), 'w') as f:
		f.write(str(FLAGS)+'\n')

	# tf placeholder definitions
	img_inp = tf.placeholder(tf.float32, shape=(None, HEIGHT, WIDTH, 4), name='img_inp')
	theta = tf.placeholder(tf.float32, shape=(None, 1), name='theta')
	img_gt = tf.placeholder(tf.float32, shape=(None, HEIGHT, WIDTH, 4), name='img_gt')

	# summary writer definitions
	tf.summary.image('input',img_inp,TF_SUMMARY_MAX_OUT)
	tf.summary.image('gt',img_gt,TF_SUMMARY_MAX_OUT)

	# build the graph
	out = multiview_net(img_inp, theta, TF_SUMMARY_MAX_OUT)
	loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(img_gt, out), 2), 3))

	# data
	train_models, training_pair_indices, val_models, val_pair_indices = get_models(DATA_DIR, NUM_VIEWS, NUM_GT, BATCH_SIZE, VAL_BATCH_SIZE)
	batches = len(training_pair_indices)/BATCH_SIZE

	# optimizer
	optim = tf.train.AdamOptimizer(FLAGS.lr, beta1=0.9).minimize(loss)

	# logging
	snapshot_folder = join(EXP_DIR,'snapshots')
	log_folder = join(EXP_DIR,'logs')
	create_folder(snapshot_folder)
	create_folder(log_folder)
	create_folder(os.path.join(snapshot_folder,'best'))

	# saver
	saver = tf.train.Saver(max_to_keep=2)

	# tf summaries
	tf.summary.scalar('loss', loss)
	merged_summ = tf.summary.merge_all()

	# create tf session
	with tf.Session() as sess:
		train_writer = tf.summary.FileWriter(log_folder+'/train',sess.graph_def)
		val_writer = tf.summary.FileWriter(log_folder+'/val',sess.graph_def)

		sess.run(tf.global_variables_initializer())

		start_epoch = 0
		max_epoch = FLAGS.max_epoch

		# load checkpoint if present
		ckpt = tf.train.get_checkpoint_state('snapshots')
		if ckpt is not None:
			print ('loading '+ckpt.model_checkpoint_path + '  ....')
			saver.restore(sess, ckpt.model_checkpoint_path)
			start_epoch = int(re.match('.*-(\d*)$', ckpt.model_checkpoint_path).group(1))

		ind = 0
		best_val_loss = 10000000
		since = time.time()

		for epoch in range(start_epoch, max_epoch):
			train_epoch_loss = 0.
			random.shuffle(training_pair_indices)

			# training
			for b in xrange(batches):
				global_step = epoch*batches+b+1
				batch_ip,batch_theta,batch_gt = fetch_batch(train_models, training_pair_indices, b, BATCH_SIZE)
				L, _, summ = sess.run([loss, optim, merged_summ], feed_dict={img_inp:batch_ip, theta:batch_theta, img_gt:batch_gt})
				train_epoch_loss += L/batches

				if global_step % 100 == 0:
					print 'Loss = {}  Iter = {}  Minibatch = {}'.format(L, global_step, b)
					train_writer.add_summary(summ, global_step)

				if global_step % 5000 == 0:
					val_epoch_loss, val_summ = get_epoch_loss(val_models, val_pair_indices)
					val_writer.add_summary(val_summ, global_step)

			# validation
			val_epoch_loss, val_summ = get_epoch_loss(val_models, val_pair_indices)

			time_elapsed = time.time() - since
			print '-'*20+' EPOCH '+str(epoch)+' '+'-'*20
			print 'Train Loss: {:.2f}  Val Loss: {:.2f}   Time:{:.2f}m {:.2f}s'.format(train_epoch_loss, val_epoch_loss, time_elapsed//60, time_elapsed%60)
			print

			saver.save(sess, os.path.join('snapshots', 'model'), global_step=epoch)

			# save best model
			if (val_epoch_loss < best_val_loss):
				saver.save(sess, os.path.join('snapshots/best', 'best'))
				os.system('cp snapshots/best/* best/')
				best_val_loss = val_epoch_loss