import logging

import cv2
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope

from neuralgym.ops.layers import resize
from neuralgym.ops.layers import *
from neuralgym.ops.loss_ops import *
from neuralgym.ops.summary_ops import *


logger = logging.getLogger()
class Aspp:
    def __init__(self):
        self.dilation_rate = [1,2,4,8]
        self.cnum=32
        # shortcut ops
        self.conv_3 = partial(tf.layers.conv2d, kernel_size=3, activation=tf.nn.elu, padding='SAME')
    def get_fuse_feature(self,x, name):
        feature_1=self.conv_3(inputs=x, filters=4 * self.cnum, strides=1, dilation_rate=self.dilation_rate[0], name=name+'_conv7_atrous_'+str(self.dilation_rate[0]))
        feature_2=self.conv_3(inputs=x, filters=4 * self.cnum, strides=1, dilation_rate=self.dilation_rate[1], name=name+'_conv7_atrous_'+str(self.dilation_rate[1]))
        feature_3=self.conv_3(inputs=x, filters=4 * self.cnum, strides=1, dilation_rate=self.dilation_rate[2], name=name+'_conv7_atrous_'+str(self.dilation_rate[2]))    
        feature_4=self.conv_3(inputs=x, filters=4 * self.cnum, strides=1, dilation_rate=self.dilation_rate[3], name=name+'_conv7_atrous_'+str(self.dilation_rate[3]))
        fuse_feature=(feature_1+feature_2+feature_3+feature_4)/4
        
        return fuse_feature
def resize_mask_like(mask, x):
    """Resize mask like shape of x.

    Args:
        mask: Original mask.
        x: To shape of x.

    Returns:
        tf.Tensor: resized mask

    """
    mask_resize = resize(
        mask, to_shape=x.get_shape().as_list()[1:3],
        func=tf.image.resize_nearest_neighbor)
    return mask_resize
def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


def flow_to_image_tf(flow, name='flow_to_image'):
    """Tensorflow ops for computing flow to image.
    """
    with tf.variable_scope(name), tf.device('/cpu:0'):
        img = tf.py_func(flow_to_image, [flow], tf.float32, stateful=False)
        img.set_shape(flow.get_shape().as_list()[0:-1]+[3])
        img = img / 127.5 - 1.
        return img
def contextual_attention(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.

    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.

    Returns:
        tf.Tensor: output

    """
    # get shapes
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.extract_image_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        mask = resize(mask, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.extract_image_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])

        # softmax to match
        yi *=  mm  # mask
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask

        offset = tf.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        flow = resize(flow, scale=rate, func=tf.image.resize_nearest_neighbor)
    return y, flow
    
def contextual_attention_max(f, b, mask=None, ksize=3, stride=1, rate=1,
                         fuse_k=3, softmax_scale=10., training=True, fuse=True):
    """ Contextual attention layer implementation.

    Contextual attention is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.

    Args:
        x: Input feature to match (foreground).
        t: Input feature for match (background).
        mask: Input mask for t, indicating patches not available.
        ksize: Kernel size for contextual attention.
        stride: Stride for extracting patches from t.
        rate: Dilation for matching.
        softmax_scale: Scaled softmax for attention.
        training: Indicating if current graph is training or inference.

    Returns:
        tf.Tensor: output

    """
    # get shapes
    raw_fs = tf.shape(f)
    raw_int_fs = f.get_shape().as_list()
    raw_int_bs = b.get_shape().as_list()
    #print(raw_fs,raw_int_fs,raw_int_bs)#[6,64,64,128]
    #exit(0)
    # extract patches from background with stride and rate
    kernel = 2*rate
    raw_w = tf.extract_image_patches(
        b, [1,kernel,kernel,1], [1,rate*stride,rate*stride,1], [1,1,1,1], padding='SAME')#
    print(raw_w.shape)
    #exit(0)    
    raw_w = tf.reshape(raw_w, [raw_int_bs[0], -1, kernel, kernel, raw_int_bs[3]])
    print(raw_w.shape)    
    raw_w = tf.transpose(raw_w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    print(raw_w.shape)
    #exit(0)     
    # downscaling foreground option: downscaling both foreground and
    # background for matching and use original background for reconstruction.
    f = resize(f, scale=1./rate, func=tf.image.resize_nearest_neighbor)#[6,32,32,128]
    b = resize(b, to_shape=[int(raw_int_bs[1]/rate), int(raw_int_bs[2]/rate)], func=tf.image.resize_nearest_neighbor)  # https://github.com/tensorflow/tensorflow/issues/11651
    if mask is not None:
        mask = resize(mask, scale=1./rate, func=tf.image.resize_nearest_neighbor)
    fs = tf.shape(f)
    int_fs = f.get_shape().as_list()
    f_groups = tf.split(f, int_fs[0], axis=0)
    #print(f.shape,b.shape,fs,int_fs,f_groups)
    #exit(0)
    # from t(H*W*C) to w(b*k*k*c*h*w)
    bs = tf.shape(b)
    int_bs = b.get_shape().as_list()
    w = tf.extract_image_patches(
        b, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    w = tf.reshape(w, [int_fs[0], -1, ksize, ksize, int_fs[3]])
    w = tf.transpose(w, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    print(w.shape)
    #exit(0)
    # process mask
    if mask is None:
        mask = tf.zeros([1, bs[1], bs[2], 1])
    m = tf.extract_image_patches(
        mask, [1,ksize,ksize,1], [1,stride,stride,1], [1,1,1,1], padding='SAME')
    m = tf.reshape(m, [1, -1, ksize, ksize, 1])
    m = tf.transpose(m, [0, 2, 3, 4, 1])  # transpose to b*k*k*c*hw
    m = m[0]
    mm = tf.cast(tf.equal(tf.reduce_mean(m, axis=[0,1,2], keep_dims=True), 0.), tf.float32)
    w_groups = tf.split(w, int_bs[0], axis=0)
    raw_w_groups = tf.split(raw_w, int_bs[0], axis=0)
    y = []
    offsets = []
    k = fuse_k
    scale = softmax_scale*100
    fuse_weight = tf.reshape(tf.eye(k), [k, k, 1, 1])
    for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
        # conv for compare
        wi = wi[0]
        wi_normed = wi / tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(wi), axis=[0,1,2])), 1e-4)
        yi = tf.nn.conv2d(xi, wi_normed, strides=[1,1,1,1], padding="SAME")

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1], bs[2]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
            yi = tf.reshape(yi, [1, fs[1]*fs[2], bs[1]*bs[2], 1])
            yi = tf.nn.conv2d(yi, fuse_weight, strides=[1,1,1,1], padding='SAME')
            yi = tf.reshape(yi, [1, fs[2], fs[1], bs[2], bs[1]])
            yi = tf.transpose(yi, [0, 2, 1, 4, 3])
        #print(yi.shape)
        yi = tf.reshape(yi, [1, fs[1], fs[2], bs[1]*bs[2]])
        #print(yi.shape)
        #exit(0)

        # softmax to match
        yi *=  mm  # mask
        #print(mm.shape)
        #exit(0)
        yi = tf.nn.softmax(yi*scale, 3)
        yi *=  mm  # mask

        offset = tf.argmax(yi, axis=3, output_type=tf.int32)
        offset = tf.stack([offset // fs[2], offset % fs[2]], axis=-1)
        # deconv for patch pasting
        # 3.1 paste center
        wi_center = raw_wi[0]
        yi = tf.nn.conv2d_transpose(yi, wi_center, tf.concat([[1], raw_fs[1:]], axis=0), strides=[1,rate,rate,1]) / 4.
        y.append(yi)
        offsets.append(offset)
    y = tf.concat(y, axis=0)
    y.set_shape(raw_int_fs)
    offsets = tf.concat(offsets, axis=0)
    offsets.set_shape(int_bs[:3] + [2])
    # case1: visualize optical flow: minus current position
    h_add = tf.tile(tf.reshape(tf.range(bs[1]), [1, bs[1], 1, 1]), [bs[0], 1, bs[2], 1])
    w_add = tf.tile(tf.reshape(tf.range(bs[2]), [1, 1, bs[2], 1]), [bs[0], bs[1], 1, 1])
    offsets = offsets - tf.concat([h_add, w_add], axis=3)
    # to flow image
    flow = flow_to_image_tf(offsets)
    # # case2: visualize which pixels are attended
    # flow = highlight_flow_tf(offsets * tf.cast(mask, tf.int32))
    if rate != 1:
        flow = resize(flow, scale=rate, func=tf.image.resize_nearest_neighbor)
    return y, flow    