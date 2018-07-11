import os
import sys
import numpy as np
import cv2
import random
from scipy import misc
from itertools import product
from os.path import join


def create_folder(folder):
	''' Creates a folder if it doesn't already exist '''
	if not os.path.exists(folder):
		os.makedirs(folder)


def verify_model(model_path, num_views, num_gt):
	''' Verifies if all necessary inputs are present '''
	gt_image_names = os.listdir(os.path.join(model_path,'gt'))
	ip_image_names = os.listdir(os.path.join(model_path,'ip'))
	if len(gt_image_names) != num_gt or len(ip_image_names) != num_views:
		print model_path
		return False
	return True


def get_models(data_dir, num_views, num_gt, batch_size, val_batch_size):
	''' Training and validation data models
	Args:
		data_dir: data directory
		num_view: number of input images (having random azimuth values)
		num_gt: number of ground truth images (having fixed azimuth values)
		batch_size: batch size during training
		val_batch_size: batch size duiring validation
	Returns:
		train_models: list of train model paths
		training_pair_indices: list of tuples of the form (train_model_path,input_image_idx,gt_image_idx)
		val_models: list of validation model paths
		val_pair_indices: list of tuples of the form (val_model_path,input_image_idx,gt_image_idx)
	'''
	all_dirs = [join(data_dir,d) for d in os.listdir(data_dir)][:10]
	print
	print len(all_dirs), ' total models.'
	all_models = []
	for i,dir in enumerate(all_dirs):
		if verify_model(dir, num_views, num_gt):
			all_models.append(dir)
		if i%500==0:
			print i, ' models verified'
	num_models = len(all_models)
	print num_models, ' models valid.'
	train_models = all_models[:int(0.8*num_models)]
	val_models = all_models[int(0.8*num_models):]
	training_pair_indices = list(product(xrange(len(train_models)),xrange(num_views),xrange(num_gt)))
	val_pair_indices = list(product(xrange(len(val_models)),xrange(num_views),xrange(num_gt)))
	batches = len(training_pair_indices)/batch_size
	print 'TRAINING: models={}  samples={}  batches={}'.format(len(train_models),len(train_models)*num_views*num_gt,batches)
	print 'VALIDATION: models={}  samples={}  batches={}'.format(len(val_models),len(val_models)*num_views*num_gt,len(val_pair_indices)/val_batch_size)
	print

	return train_models, training_pair_indices, val_models, val_pair_indices


def fetch_batch(models, indices, batch_num, batch_size):
	''' Fetches a batch for training
	Args:
		models: train model paths list
		indices: indices at which we want to query the models list
		batch_num: iteration batch number
		batch_size: BATCH_SIZE for trai, VAL_BATCH_SIE for val
	Returns:
		batch_ip: input image numpy array (batch_size,HEIGHT,WIDTH,3)
		batch_theta: theta for the different inputs (batch_size,1)
		batch_gt: ground truth numpy array (batch_size,HEIGHT,WIDTH,3)
	 '''
	batch_ip = []
	batch_theta = []
	batch_gt = []

	iteration = 0
	for ind in indices[batch_num*batch_size:batch_num*batch_size+batch_size+iteration]:
		model_path = models[ind[0]]
		gt_image_names = os.listdir(os.path.join(model_path,'gt'))
		
		try:
			gt_image_name = gt_image_names[ind[2]]
			gt_image = misc.imread(os.path.join(model_path,'gt',gt_image_name))
			gt_image = gt_image.astype('float32')/255.
			theta = int(gt_image_name.split('_')[2][1:])
			ip_image_names = os.listdir(os.path.join(model_path,'ip'))			
			try:
				ip_image_name = ip_image_names[ind[1]]
				ip_image = misc.imread(os.path.join(model_path,'ip',ip_image_name))
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


