import tensorflow as tf
from tf_utils import _conv_2d,_fc,_conv_2d_transpose


def multiview_net(img_inp, theta_inp, tf_summary_max_out):
	# ========== encoder ==========
	x=img_inp
	#128 128
	x=_conv_2d(x,32,[3,3],1)
	x=_conv_2d(x,32,[3,3],1)
	x=_conv_2d(x,64,[3,3],2)
	#64 64
	x=_conv_2d(x,64,[3,3],1)
	x=_conv_2d(x,64,[3,3],1)
	x=_conv_2d(x,128,[3,3],2)
	#32 32
	x=_conv_2d(x,128,[3,3],1)
	x=_conv_2d(x,128,[3,3],1)
	x=_conv_2d(x,256,[3,3],2)
	#16 16
	x=_conv_2d(x,256,[3,3],1)
	x=_conv_2d(x,256,[3,3],1)
	x=_conv_2d(x,512,[3,3],2)
	#8 8
	x=_conv_2d(x,512,[3,3],1)
	x=_conv_2d(x,512,[3,3],1)
	x=_conv_2d(x,512,[3,3],1)
	x=_conv_2d(x,512,[5,5],2)
	x=_fc(x,128)
	# ========== decoder ==========
	x = tf.concat([x, theta_inp], axis=1)
	x = _fc(x,512)
	x = _fc(x,1024)
	x = _fc(x,8192)
	x = tf.reshape(x, [-1,8,8,128])
	x = _conv_2d_transpose(x,256,[5,5],[16,16],2)
	#16 16
	x = _conv_2d(x,256,[3,3],1)
	x = _conv_2d(x,256,[3,3],1)
	x = _conv_2d_transpose(x,128,[5,5],[32,32],2)
	#32 32
	x = _conv_2d(x,128,[3,3],1)
	x = _conv_2d(x,128,[3,3],1)
	x = _conv_2d_transpose(x,64,[5,5],[64,64],2)
	#64 64
	x = _conv_2d(x,64,[3,3],1)
	x = _conv_2d(x,64,[3,3],1)
	x = _conv_2d_transpose(x,32,[5,5],[128,128],2)
	#128 128
	x = _conv_2d(x,32,[3,3],1)
	x = _conv_2d(x,4,[3,3],1,bn=False)
	x = tf.nn.sigmoid(x)
	tf.summary.image('output',x,tf_summary_max_out)
	return x

