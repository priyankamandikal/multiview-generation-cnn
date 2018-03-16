import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import tflearn
import random
from scipy import misc
import re
from itertools import product
import time

# seed for reproducibility
random.seed(1024)

# User args
GPU_ID = str(sys.argv[1])
DATA_DIR = str(sys.argv[2])
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_ID

# TRAINING PARAMETERS
BATCH_SIZE = 32
VAL_BATCH_SIZE = 32
HEIGHT = 192
WIDTH = 256
NUM_VIEWS = 12
NUM_GT = 8
TF_SUMMARY_MAX_OUT = 12


''' Graph Definition
	A simple auto-encoder network conditioned on theta to take in an image of an object 
	and output an image of that object at the desired viewpoint according to theta.
Inputs
	img_inp --> input image tensor of size (BATCH_SIZE,192,256,3)
	theta_inp --> input azimuth angle for desired output viewpoint (BATCH_SIZE,1)
Returns
	out --> output image normalized between -1 to 1 (BATCH_SIZE,192,256,3)
'''
def buildgraph(img_inp, theta_inp):
	x = img_inp
	theta = theta_inp

	# image processing
#192 256
	x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x0=x
	x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
#96 128
	x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x1=x
	x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
#48 64
	x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x2=x
	x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
#24 32
	x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x3=x
	x=tflearn.layers.conv.conv_2d(x,256,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
#12 16
	x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x4=x
	x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
#6 8
	x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d(x,512,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x5=x
	x=tflearn.layers.conv.conv_2d(x,512,(5,5),strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
#3 4
	x = tflearn.layers.core.fully_connected(x,2048,activation='relu',weight_decay=1e-3,regularizer='L2')
	x = tflearn.layers.core.fully_connected(x,1024,activation='relu',weight_decay=1e-3,regularizer='L2')
# 1024

	# angle processing
	theta = tflearn.layers.core.fully_connected(theta,64,activation='relu',weight_decay=1e-3,regularizer='L2')
	theta = tflearn.layers.core.fully_connected(theta,64,activation='relu',weight_decay=1e-3,regularizer='L2')
	theta = tflearn.layers.core.fully_connected(theta,64,activation='relu',weight_decay=1e-3,regularizer='L2')

	concatenated = tf.concat([x, theta],axis=1)

	# joint processing
	x = tflearn.layers.core.fully_connected(concatenated,8192,activation='relu',weight_decay=1e-3,regularizer='L2')
	x = tflearn.layers.core.fully_connected(x,8192,activation='relu',weight_decay=1e-3,regularizer='L2')
	x = tflearn.layers.core.fully_connected(x,6144,activation='relu',weight_decay=1e-3,regularizer='L2')
	x = tf.reshape(x, [-1, 3, 4, 512])
# 3 4
	############################################## ENCODER ENDS #############################################
	#########################################################################################################
	############################################# DECODER BEGINS ############################################
	x=tflearn.layers.conv.conv_2d_transpose(x,256,[5,5],[6,8],strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
#6 8
	x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d(x,256,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d_transpose(x,128,[5,5],[12,16],strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
#12 16
	x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d(x,128,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d_transpose(x,64,[5,5],[24,32],strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
#24 32
	x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d(x,64,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d_transpose(x,32,[5,5],[48,64],strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
#48 64
	x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d(x,32,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d_transpose(x,16,[5,5],[96,128],strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
#96 128
	x=tflearn.layers.conv.conv_2d(x,16,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d_transpose(x,8,[5,5],[192,256],strides=2,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
#192 256
	x=tflearn.layers.conv.conv_2d(x,8,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')
	x=tflearn.layers.normalization.batch_normalization(x)
	x=tflearn.layers.conv.conv_2d(x,4,(3,3),strides=1,activation='relu',weight_decay=1e-5,regularizer='L2')

	out = tf.nn.sigmoid(x)
	tf.summary.image('output',x,TF_SUMMARY_MAX_OUT)
	return x


''' Creates a folder if it doens't already exist '''
def create_folder(folder):
	if not os.path.exists(folder):
		os.makedirs(folder)


''' Verifies if all necessary inputs are present '''
def verify_model(model_path):

	gt_image_names = os.listdir(os.path.join(model_path,'gt'))
	ip_image_names = os.listdir(os.path.join(model_path,'ip'))
	if len(gt_image_names) != NUM_GT or len(ip_image_names) != NUM_VIEWS:
		print model_path
		return False
	return True


''' Fetches a batch for training
Inputs
	models --> train model paths list
	indices --> indices at which we want to query the models list
	batch_num --> iteration batch number
	batch_size --> BATCH_SIZE for trai, VAL_BATCH_SIE for val
Returns
	batch_ip --> input image numpy array (batch_size,HEIGHT,WIDTH,3)
	batch_theta --> theta for the different inputs (batch_size,1)
	batch_gt --> ground truth numpy array (batch_size,HEIGHT,WIDTH,3)
 '''
def fetch_batch(models, indices, batch_num, batch_size):
	batch_ip = []
	batch_theta = []
	batch_gt = []

	iteration = 0
	for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size+iteration]:

		model_path = models[ind[0]]

		gt_image_names = os.listdir(os.path.join(model_path,'gt'))
		
		try:
			gt_image_name = gt_image_names[ind[2]]
			gt_image = misc.imresize(misc.imread(os.path.join(model_path,'gt',gt_image_name)), (192,256), 'bicubic')
			gt_image = gt_image.astype('float32')/255.

			theta = int(gt_image_name.split('_')[2][1:])

			ip_image_names = os.listdir(os.path.join(model_path,'ip'))
			
			try:
				ip_image_name = ip_image_names[ind[1]]
				ip_image = misc.imresize(misc.imread(os.path.join(model_path,'ip',ip_image_name)), (192,256), 'bicubic')
				ip_image = ip_image.astype('float32')/255.
				batch_ip.append(ip_image)
				batch_theta.append(theta)
				batch_gt.append(gt_image)
			except:
				print (model_path)
				pass

		except:
			print (model_path)
			pass

	batch_ip = np.array(batch_ip)
	batch_theta = np.array(batch_theta, dtype=np.float32).reshape(-1,1)
	batch_gt = np.array(batch_gt)

	return batch_ip, batch_theta, batch_gt


''' Validation loss calculation 
Inputs
	val_models --> val model paths list
	val_pair_indices --> indices at which we want to query the models list
Returns
	val_loss --> validation loss (L2 between GT and predicted images)
	summ --> val summary writer
'''
def get_epoch_loss(val_models, val_pair_indices):

	batches = len(val_pair_indices)/VAL_BATCH_SIZE
	val_loss = 0.
	batches = 500
	for b in xrange(batches):
		batch_ip, batch_theta, batch_gt = fetch_batch(val_models, val_pair_indices, b, VAL_BATCH_SIZE)
		L, summ = sess.run([loss, merged_summ], feed_dict={img_inp:batch_ip, theta:batch_theta, img_gt:batch_gt})
		val_loss += L/batches
	
	return val_loss, summ


if __name__=='__main__':

	# tf placeholder definitions
	img_inp = tf.placeholder(tf.float32, shape=(None, HEIGHT, WIDTH, 4), name='img_inp')
	theta = tf.placeholder(tf.float32, shape=(None, 1), name='theta')
	img_gt = tf.placeholder(tf.float32, shape=(None, HEIGHT, WIDTH, 4), name='img_gt')

	# summary writer definitions
	tf.summary.image('input',img_inp,TF_SUMMARY_MAX_OUT)
	tf.summary.image('gt',img_gt,TF_SUMMARY_MAX_OUT)

	# build the graph
	out = buildgraph(img_inp, theta)
	loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(img_gt, out), 2), 3))

	# data loading ops
	all_dirs = [DATA_DIR + d for d in os.listdir(DATA_DIR)]
	print
	print len(all_dirs), ' total models.'
	all_models = []
	for i,dir in enumerate(all_dirs):
		if verify_model(dir):
			all_models.append(dir)
		if i%500==0:
			print i, ' models verified'

	num_models = len(all_models)
	print num_models, ' models valid.'
	train_models = all_models[:int(0.8*num_models)]
	val_models = all_models[int(0.8*num_models):]
	training_pair_indices = list(product(xrange(len(train_models)),xrange(NUM_VIEWS),xrange(NUM_GT)))
	val_pair_indices = list(product(xrange(len(val_models)),xrange(NUM_VIEWS),xrange(NUM_GT)))
	batches = len(training_pair_indices)/BATCH_SIZE
	print 'TRAINING: models={}  samples={}  batches={}'.format(len(train_models),len(train_models)*NUM_VIEWS*NUM_GT,batches)
	print 'VALIDATION: models={}  samples={}  batches={}'.format(len(val_models),len(val_models)*NUM_VIEWS*NUM_GT,len(val_pair_indices)/VAL_BATCH_SIZE)
	print

	# optimizer
	optim = tf.train.AdamOptimizer(0.0001, beta1=0.9).minimize(loss)

	# training params
	start_epoch = 0
	max_epoch = 20
	snapshot_folder = 'snapshots'
	saver = tf.train.Saver(max_to_keep=2)
	log_folder = "logs_mult_view"
	tf.summary.scalar('loss', loss)
	merged_summ = tf.summary.merge_all()

	create_folder(snapshot_folder)
	create_folder(log_folder)
	create_folder(os.path.join(snapshot_folder,'best'))

	# create tf session
	with tf.Session() as sess:
		train_writer = tf.summary.FileWriter(log_folder+'/train',sess.graph_def)
		val_writer = tf.summary.FileWriter(log_folder+'/val',sess.graph_def)


		sess.run(tf.global_variables_initializer())

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