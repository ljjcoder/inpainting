import tensorflow as tf
from net.ops import random_bbox, bbox2mask, local_patch,bbox2mask_center
from net.ops import priority_loss_mask
from net.ops import id_mrf_reg,littilewave
from net.ops import gan_wgan_loss, gradients_penalty, random_interpolates
from net.ops import free_form_mask_tf
from util.util import f2uint
from functools import partial

class GMCNNModel:
    def __init__(self):
        self.config = None

        # shortcut ops
        self.conv7 = partial(tf.layers.conv2d, kernel_size=7, activation=tf.nn.elu, padding='SAME')
        self.conv5 = partial(tf.layers.conv2d, kernel_size=5, activation=tf.nn.elu, padding='SAME')
        self.conv3 = partial(tf.layers.conv2d, kernel_size=3, activation=tf.nn.elu, padding='SAME')
        self.conv5_ds = partial(tf.layers.conv2d, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='SAME')

    def build_generator(self, x, mask, reuse=False, name='inpaint_net'):
        xshape = x.get_shape().as_list()
        xh, xw = xshape[1], xshape[2]
        ones_x = tf.ones_like(x)[:, :, :, 0:1]
        x_w_mask = tf.concat([x, ones_x, ones_x * mask], axis=3)

        # network with three branches
        cnum = self.config.g_cnum
        b_names = ['b1', 'b2', 'b3', 'merge']

        conv_7 = self.conv7
        conv_5 = self.conv5
        conv_3 = self.conv3
        with tf.variable_scope(name, reuse=reuse):
            # branch 1
            x = conv_7(inputs=x_w_mask, filters=cnum, strides=1, name=b_names[0] + 'conv1')
            x = conv_7(inputs=x, filters=2*cnum, strides=2, name=b_names[0] + 'conv2_downsample')
            x = conv_7(inputs=x, filters=2*cnum, strides=1, name=b_names[0] + 'conv3')
            x = conv_7(inputs=x, filters=4*cnum, strides=2, name=b_names[0] + 'conv4_downsample')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, name=b_names[0] + 'conv5')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, name=b_names[0] + 'conv6')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, dilation_rate=2, name=b_names[0] + 'conv7_atrous')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, dilation_rate=4, name=b_names[0] + 'conv8_atrous')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, dilation_rate=8, name=b_names[0] + 'conv9_atrous')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, dilation_rate=16, name=b_names[0] + 'conv10_atrous')
            if cnum > 32:
                x = conv_7(inputs=x, filters=4 * cnum, strides=1, dilation_rate=32, name=b_names[0] + 'conv11_atrous')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, name=b_names[0] + 'conv11')
            x = conv_7(inputs=x, filters=4*cnum, strides=1, name=b_names[0] + 'conv12')
            x_b1 = tf.image.resize_bilinear(x, [xh, xw], align_corners=True)

            # branch 2
            x = conv_5(inputs=x_w_mask, filters=cnum, strides=1, name=b_names[1] + 'conv1')
            x = conv_5(inputs=x, filters=2 * cnum, strides=2, name=b_names[1] + 'conv2_downsample')
            x = conv_5(inputs=x, filters=2 * cnum, strides=1, name=b_names[1] + 'conv3')
            x = conv_5(inputs=x, filters=4 * cnum, strides=2, name=b_names[1] + 'conv4_downsample')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, name=b_names[1] + 'conv5')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, name=b_names[1] + 'conv6')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, dilation_rate=2, name=b_names[1] + 'conv7_atrous')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, dilation_rate=4, name=b_names[1] + 'conv8_atrous')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, dilation_rate=8, name=b_names[1] + 'conv9_atrous')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, dilation_rate=16, name=b_names[1] + 'conv10_atrous')
            if cnum > 32:
                x = conv_5(inputs=x, filters=4 * cnum, strides=1, dilation_rate=32, name=b_names[1] + 'conv11_atrous')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, name=b_names[1] + 'conv11')
            x = conv_5(inputs=x, filters=4 * cnum, strides=1, name=b_names[1] + 'conv12')
            x = tf.image.resize_nearest_neighbor(x, [xh//2, xw//2], align_corners=True)
            with tf.variable_scope(b_names[1] + 'conv13_upsample'):
                x = conv_3(inputs=x, filters=2 * cnum, strides=1, name=b_names[1] + 'conv13_upsample_conv')
            x = conv_5(inputs=x, filters=2 * cnum, strides=1, name=b_names[1] + 'conv14')
            x_b2 = tf.image.resize_bilinear(x, [xh, xw], align_corners=True)

            # branch 3
            x = conv_5(inputs=x_w_mask, filters=cnum, strides=1, name=b_names[2] + 'conv1')
            x = conv_3(inputs=x, filters=2 * cnum, strides=2, name=b_names[2] + 'conv2_downsample')
            x = conv_3(inputs=x, filters=2 * cnum, strides=1, name=b_names[2] + 'conv3')
            x = conv_3(inputs=x, filters=4 * cnum, strides=2, name=b_names[2] + 'conv4_downsample')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, name=b_names[2] + 'conv5')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, name=b_names[2] + 'conv6')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, dilation_rate=2, name=b_names[2] + 'conv7_atrous')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, dilation_rate=4, name=b_names[2] + 'conv8_atrous')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, dilation_rate=8, name=b_names[2] + 'conv9_atrous')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, dilation_rate=16, name=b_names[2] + 'conv10_atrous')
            if cnum > 32:
                x = conv_3(inputs=x, filters=4 * cnum, strides=1, dilation_rate=32, name=b_names[2] + 'conv11_atrous')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, name=b_names[2] + 'conv11')
            x = conv_3(inputs=x, filters=4 * cnum, strides=1, name=b_names[2] + 'conv12')
            x = tf.image.resize_nearest_neighbor(x, [xh // 2, xw // 2], align_corners=True)
            with tf.variable_scope(b_names[2] + 'conv13_upsample'):
                x = conv_3(inputs=x, filters=2 * cnum, strides=1, name=b_names[2] + 'conv13_upsample_conv')
            x = conv_3(inputs=x, filters=2 * cnum, strides=1, name=b_names[2] + 'conv14')
            x = tf.image.resize_nearest_neighbor(x, [xh, xw], align_corners=True)
            with tf.variable_scope(b_names[2] + 'conv15_upsample'):
                x = conv_3(inputs=x, filters=cnum, strides=1, name=b_names[2] + 'conv15_upsample_conv')
            x_b3 = conv_3(inputs=x, filters=cnum//2, strides=1, name=b_names[2] + 'conv16')

            x_merge = tf.concat([x_b1, x_b2, x_b3], axis=3)

            x = conv_3(inputs=x_merge, filters=cnum // 2, strides=1, name=b_names[3] + 'conv17')
            x = tf.layers.conv2d(inputs=x, kernel_size=3, filters=3, strides=1, activation=None, padding='SAME',
                                 name=b_names[3] + 'conv18')
            x = tf.clip_by_value(x, -1., 1.)
        return x


    def wgan_patch_discriminator(self, x, mask, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('discriminator_local', reuse=reuse):
            h, w = mask.get_shape().as_list()[1:3]
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            #print(x)
            x = self.conv5_ds(x, filters=cnum*2, name='conv2')
            #print(x)            
            x = self.conv5_ds(x, filters=cnum*4, name='conv3')
            #print(x)            
            x = self.conv5_ds(x, filters=cnum*8, name='conv4')
            print(x)            
            x = tf.layers.conv2d(x, kernel_size=5, strides=2, filters=1, activation=None, name='conv5', padding='SAME')
            print(x,'lllll')
            #exit(0)

            mask = tf.contrib.layers.max_pool2d(mask, 2, padding='SAME')
            mask = tf.contrib.layers.max_pool2d(mask, 2, padding='SAME')
            mask = tf.contrib.layers.max_pool2d(mask, 2, padding='SAME')
            mask = tf.contrib.layers.max_pool2d(mask, 2, padding='SAME')
            mask = tf.contrib.layers.max_pool2d(mask, 2, padding='SAME')

            x = x * mask
            print(x,mask)
            #exit(0)            
            x = tf.reduce_sum(x, axis=[1, 2, 3]) / tf.reduce_sum(mask, axis=[1, 2, 3])
            mask_local = tf.image.resize_nearest_neighbor(mask, [h, w], align_corners=True)
            return x, mask_local


    def wgan_local_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_local', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 8, name='conv4')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv5')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv6')

            x = tf.layers.flatten(x, name='flatten')
            return x

    def wgan_global_discriminator(self, x, d_cnum, reuse=False):
        cnum = d_cnum
        with tf.variable_scope('disc_global', reuse=reuse):
            x = self.conv5_ds(x, filters=cnum, name='conv1')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv2')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv3')
            x = self.conv5_ds(x, filters=cnum * 8, name='conv4')
            x = self.conv5_ds(x, filters=cnum * 4, name='conv5')
            x = self.conv5_ds(x, filters=cnum * 2, name='conv6')
            x = tf.layers.flatten(x, name='flatten')
            #print(x)
            #exit(0)
            return x

    def wgan_discriminator(self, batch_local, batch_global, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dlocal = self.wgan_local_discriminator(batch_local, d_cnum, reuse=reuse)
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_local, dout_global
            
    def wgan_discriminator_bord(self, batch_local, batch_global, d_cnum, reuse=False):
        with tf.variable_scope('discriminator_bord', reuse=reuse):
            dlocal = self.wgan_local_discriminator(batch_local, d_cnum, reuse=reuse)
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_local = tf.layers.dense(dlocal, 1, name='dout_local_fc')
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            return dout_local, dout_global
            
    def wgan_mask_discriminator(self, batch_global, mask, d_cnum, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            #print(dout_global)
            #exit(0)
            dout_local, mask_local = self.wgan_patch_discriminator(batch_global, mask, d_cnum, reuse=reuse)
            #print(dout_local, mask_local)
            #exit(0)
        return dout_local, dout_global, mask_local
    def wgan_mask_discriminator_bord(self, batch_global, mask, d_cnum, reuse=False):
        with tf.variable_scope('discriminator_bord', reuse=reuse):
            dglobal = self.wgan_global_discriminator(batch_global, d_cnum, reuse=reuse)
            dout_global = tf.layers.dense(dglobal, 1, name='dout_global_fc')
            dout_local, mask_local = self.wgan_patch_discriminator(batch_global, mask, d_cnum, reuse=reuse)
            
        return dout_local, dout_global, mask_local
        
    def build_net(self, batch_data, config, summary=True, reuse=False):
        self.config = config
        batch_pos = batch_data / 127.5 - 1.
        # generate mask, 1 represents masked point
        if config.mask_type == 'rect':
            bbox = random_bbox(config)
            mask,h,w = bbox2mask(bbox, config, name='mask_c')
            mask_center_small= bbox2mask_center(bbox,h,w, config, name='mask_center_small')
            mask_center= bbox2mask_center(bbox,h-8,w-8, config, name='mask_center')            
            mask_bord=mask-mask_center_small
        else:
            mask,mask_center = free_form_mask_tf(parts=4, im_size=(config.img_shapes[0], config.img_shapes[1]),
                                     maxBrushWidth=40, maxLength=80, maxVertex=20)# changed by ljj
            mask_bord=mask-mask_center                                                 
        batch_incomplete = batch_pos * (1. - mask)
        mask_priority = priority_loss_mask(mask)
        batch_predicted = self.build_generator(batch_incomplete, mask, reuse=reuse)

        losses = {}
        # apply mask and complete image
        batch_complete = batch_predicted * mask + batch_incomplete * (1. - mask)
        #added by ljj
        batch_complete_center = batch_predicted * mask_center + batch_pos * (1. - mask_center)  
        #batch_complete_center = batch_predicted * mask_center + batch_incomplete * (1. - mask_center)         
        batch_complete_bord = batch_predicted * mask_bord + batch_pos * (1. - mask_bord)         
        if config.mask_type == 'rect':
            # local patches
            local_patch_batch_pos = local_patch(batch_pos, bbox)
            local_patch_batch_complete = local_patch(batch_complete, bbox)
            #added by ljj
            local_patch_batch_complete_center = local_patch(batch_complete_center, bbox)  
            local_patch_batch_complete_bord = local_patch(batch_complete_bord, bbox)    
            #added end                
            local_patch_mask = local_patch(mask, bbox)
            local_patch_batch_pred = local_patch(batch_predicted, bbox)
            mask_priority = local_patch(mask_priority, bbox)
        else:
            local_patch_batch_pos = batch_pos
            local_patch_batch_complete = batch_complete
            local_patch_batch_pred = batch_predicted
            #added by ljj
            local_patch_batch_complete_center = batch_complete_center
            local_patch_batch_complete_bord = batch_complete_bord            
            #local_patch_batch_pred = batch_predicted
            #added end            

        if config.pretrain_network:
            print('Pretrain the whole net with only reconstruction loss.')

        if not config.pretrain_network:
            config.feat_style_layers = {'conv3_2': 1.0, 'conv4_2': 1.0}
            config.feat_content_layers = {'conv4_2': 1.0}

            config.mrf_style_w = 1.0
            config.mrf_content_w = 1.0

            ID_MRF_loss = id_mrf_reg(local_patch_batch_pred, local_patch_batch_pos, config)
            # ID_MRF_loss = id_mrf_reg(batch_predicted, batch_pos, config)

            losses['ID_MRF_loss'] = ID_MRF_loss
            tf.summary.scalar('losses/ID_MRF_loss', losses['ID_MRF_loss'])

        pretrain_l1_alpha = config.pretrain_l1_alpha
        losses['l1_loss'] = \
            pretrain_l1_alpha * tf.reduce_mean(tf.abs(local_patch_batch_pos - local_patch_batch_pred) * mask_priority)
        if not config.pretrain_network:
            losses['l1_loss'] += tf.reduce_mean(ID_MRF_loss * config.mrf_alpha)
        losses['ae_loss'] = pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_pos - batch_predicted) * (1. - mask))
        if not config.pretrain_network:
            losses['ae_loss'] += pretrain_l1_alpha * tf.reduce_mean(tf.abs(batch_pos - batch_predicted) * (1. - mask))
        losses['ae_loss'] /= tf.reduce_mean(1. - mask)
        #added by ljj
        a=littilewave(batch_complete,4)	
        b=littilewave(batch_pos,4)
        center_LW=littilewave(batch_complete_center,4)
        bord_LW=littilewave(batch_complete_bord,4)        
        #batch_complete_center
        #c=littilewave(batch_complete_x1,4)	
        self.harr_loss_x2=0	
        #self.harr_loss_x1=0
        #print('pppppppppppppppppp')
        #exit(0)        
        for i in range(len(a)):
            #if (i%4)==0:
                #print('0000')
                #exit(0)
            #if ((i%4)==0)&(i>5):
                #continue	
            self.harr_loss_x2=self.harr_loss_x2+tf.reduce_mean(tf.square(a[i]-b[i]))#+tf.reduce_mean(tf.square(center_LW[i]-b[i]))+tf.reduce_mean(tf.square(bord_LW[i]-b[i]))
            #self.harr_loss_x1=self.harr_loss_x1+tf.reduce_mean(tf.square(c[i]-b[i]))
        #added end 
        if summary:
            viz_img = tf.concat([batch_pos, batch_incomplete, batch_predicted, batch_pos * (1. - mask_center)], axis=2)[:, :, :, ::-1]
            tf.summary.image('gt__degraded__predicted__completed', f2uint(viz_img))
            tf.summary.scalar('losses/l1_loss', losses['l1_loss'])
            tf.summary.scalar('losses/ae_loss', losses['ae_loss'])

        # gan
        batch_pos_neg = tf.concat([batch_pos, batch_complete], axis=0)
        batch_pos_neg_center = tf.concat([batch_pos, batch_complete_center], axis=0)
        batch_pos_neg_bord = tf.concat([batch_pos, batch_complete_bord], axis=0)        
        if config.mask_type == 'rect':
            # local deterministic patch
            local_patch_batch_pos_neg_center = tf.concat([local_patch_batch_pos, local_patch_batch_complete_center], 0)
            # wgan with gradient penalty
            pos_neg_local_center, pos_neg_global_center = self.wgan_discriminator(local_patch_batch_pos_neg_center,
                                                                    batch_pos_neg_center, config.d_cnum, reuse=reuse)
            # local deterministic patch
            local_patch_batch_pos_neg_bord = tf.concat([local_patch_batch_pos, local_patch_batch_complete_bord], 0)
            # wgan with gradient penalty
            pos_neg_local_bord, pos_neg_global_bord = self.wgan_discriminator_bord(local_patch_batch_pos_neg_bord,
                                                                    batch_pos_neg_bord, config.d_cnum, reuse=reuse)                                                                    
        else:
            pos_neg_local_center, pos_neg_global_center, mask_local_center = self.wgan_mask_discriminator(batch_pos_neg_center,
                                                                                     mask_center, config.d_cnum, reuse=reuse)
            pos_neg_local_bord, pos_neg_global_bord, mask_local_bord = self.wgan_mask_discriminator_bord(batch_pos_neg_bord,
                                                                                     mask_bord, config.d_cnum, reuse=reuse)                                                                                     
        pos_local_center, neg_local_center = tf.split(pos_neg_local_center, 2)
        pos_global_center, neg_global_center = tf.split(pos_neg_global_center, 2)
        pos_local_bord, neg_local_bord = tf.split(pos_neg_local_bord, 2)
        pos_global_bord, neg_global_bord = tf.split(pos_neg_global_bord, 2)        
        # wgan loss
        global_wgan_loss_alpha = 1.0
        g_loss_local_center, d_loss_local_center = gan_wgan_loss(pos_local_center, neg_local_center, name='gan/local_gan_center')
        g_loss_global_center, d_loss_global_center = gan_wgan_loss(pos_global_center, neg_global_center, name='gan/global_gan_center')
        g_loss_local_bord, d_loss_local_bord = gan_wgan_loss(pos_local_bord, neg_local_bord, name='gan/local_gan_bord')
        g_loss_global_bord, d_loss_global_bord = gan_wgan_loss(pos_global_bord, neg_global_bord, name='gan/global_gan_bord')        
        losses['g_loss'] = global_wgan_loss_alpha * (g_loss_global_center+g_loss_global_bord) + g_loss_local_center+g_loss_local_bord
        losses['d_loss'] = d_loss_global_center + d_loss_local_center+d_loss_global_bord + d_loss_local_bord
        #debug
        #losses['g_loss'] = global_wgan_loss_alpha * (g_loss_global_bord) + g_loss_local_bord
        #losses['d_loss'] = d_loss_global_bord + d_loss_local_bord
            #end
        # gp
        interpolates_global_center = random_interpolates(batch_pos, batch_complete_center)
        interpolates_global_bord = random_interpolates(batch_pos, batch_complete_bord)        
        if config.mask_type == 'rect':
            interpolates_local_center = random_interpolates(local_patch_batch_pos, local_patch_batch_complete_center)
            interpolates_local_bord = random_interpolates(local_patch_batch_pos, local_patch_batch_complete_bord)           
            dout_local_center, dout_global_center = self.wgan_discriminator(
                interpolates_local_center, interpolates_global_center, config.d_cnum, reuse=True)
            dout_local_bord, dout_global_bord = self.wgan_discriminator_bord(
                interpolates_local_bord, interpolates_global_bord, config.d_cnum, reuse=True)                
        else:
            interpolates_local_center = interpolates_global_center
            dout_local_center, dout_global_center, _ = self.wgan_mask_discriminator(interpolates_global_center, mask_center, config.d_cnum, reuse=True)
            interpolates_local_bord = interpolates_global_bord
            dout_local_bord, dout_global_bord, _ = self.wgan_mask_discriminator_bord(interpolates_global_bord, mask_bord, config.d_cnum, reuse=True)
        # apply penalty
        if config.mask_type == 'rect':
            penalty_local_center = gradients_penalty(interpolates_local_center, dout_local_center, mask=local_patch_mask)
            penalty_local_bord = gradients_penalty(interpolates_local_bord, dout_local_bord, mask=local_patch_mask)            
        else:
            penalty_local_center = gradients_penalty(interpolates_local_center, dout_local_center, mask=mask_center)
            penalty_local_bord = gradients_penalty(interpolates_local_bord, dout_local_bord, mask=mask_bord)            
        penalty_global_center = gradients_penalty(interpolates_global_center, dout_global_center, mask=mask_center)
        penalty_global_bord = gradients_penalty(interpolates_global_bord, dout_global_bord, mask=mask_bord)        
        losses['gp_loss'] = config.wgan_gp_lambda * (penalty_local_center + penalty_global_center)+config.wgan_gp_lambda * (penalty_local_bord + penalty_global_bord)
        losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
        #debug
        #losses['gp_loss'] = config.wgan_gp_lambda * (penalty_local_bord + penalty_global_bord)
        #losses['d_loss'] = losses['d_loss'] + losses['gp_loss']
        #end
        if summary and not config.pretrain_network:
            tf.summary.scalar('convergence/d_loss', losses['d_loss'])
            tf.summary.scalar('convergence/local_d_loss', d_loss_local_bord)
            tf.summary.scalar('convergence/global_d_loss', d_loss_global_bord)
            tf.summary.scalar('gan_wgan_loss/gp_loss', losses['gp_loss'])
            tf.summary.scalar('gan_wgan_loss/gp_penalty_local', penalty_local_bord)
            tf.summary.scalar('gan_wgan_loss/gp_penalty_global', penalty_global_bord)

        if config.pretrain_network:
            losses['g_loss'] = 0
        else:
            losses['g_loss'] = config.gan_loss_alpha * losses['g_loss']
        losses['g_loss'] += config.l1_loss_alpha * losses['l1_loss']#+self.harr_loss_x2 #remove self.harr_loss_x2 to ori code
        ##

        print('Set L1_LOSS_ALPHA to %f' % config.l1_loss_alpha)
        print('Set GAN_LOSS_ALPHA to %f' % config.gan_loss_alpha)

        losses['g_loss'] += config.ae_loss_alpha * losses['ae_loss']
        print('Set AE_LOSS_ALPHA to %f' % config.ae_loss_alpha)
        g_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'inpaint_net')
        d_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
        return g_vars, d_vars, losses,mask,mask_center

    def evaluate(self, im, mask, config, reuse=False):
        # generate mask, 1 represents masked point
        self.config = config
        im = im / 127.5 - 1
        im = im * (1 - mask)
        # inpaint
        batch_predict = self.build_generator(im, mask, reuse=reuse)
        # apply mask and reconstruct
        batch_complete = batch_predict * mask + im * (1 - mask)
        return batch_complete
