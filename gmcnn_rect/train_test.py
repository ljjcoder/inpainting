import os
import numpy as np
import cv2
import os
import subprocess
import glob
from util.util import generate_mask_rect, generate_mask_stroke
import tensorflow as tf
from net.network_test import GMCNNModel
#from net.network_ori import GMCNNModel
from data.data import DataLoader
from options.train_options import TrainOptions

config = TrainOptions().parse()
#if 'rec_mld_random' in config.tensorboard_folder:
    #from net.network_randdom_center import GMCNNModel
    
   # print('rec_mld_random')
#elif 'rec_mld_dc' in config.tensorboard_folder:
    #from net.network_dc import GMCNNModel
    
    #print('rec_mld_dc') 
   # exit(0)    
#else:
    #from net.network import GMCNNModel    
    #print('net.network') 
     
#exit(0)
model = GMCNNModel()
if os.path.isfile('/home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_1000/'):
    pathfile = open(config.dataset_path, 'rt').read().splitlines()
elif os.path.isdir('/home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_1000/'):
    pathfile = glob.glob(os.path.join('/home/lijunjie/generative_inpainting-master_global_local/data/places2_gt_1000/', '*.png'))
else:
    print('Invalid testing data file/folder path.')
    exit(1)
total_number = len(pathfile)
test_num = total_number
print('The total number of testing images is {}, and we take {} for test.'.format(total_number, test_num))
os.system('cp ./train_test.py ' +config.tensorboard_folder + 'train_test.py.backup')
os.system('cp ./net/network_ori.py '+config.tensorboard_folder + 'network_ori.py.backup')    
os.system('cp ./net/network_test.py '+config.tensorboard_folder + 'network_test.py.backup')
# training data
# print(config.img_shapes)
dataLoader = DataLoader(filename=config.dataset_path, batch_size=config.batch_size,
                        im_size=config.img_shapes)
images = dataLoader.next()[:, :, :, ::-1] # input BRG images
#dataLoader_test = DataLoader(filename=config.dataset_path, batch_size=config.batch_size,
                        #im_size=config.img_shapes)
images = dataLoader.next()[:, :, :, ::-1] # input BRG images
g_vars, d_vars, losses = model.build_net(images, config=config)

lr = tf.get_variable(
    'lr', shape=[], trainable=False,
    initializer=tf.constant_initializer(config.lr))

g_optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
d_optimizer = g_optimizer

g_train_op = g_optimizer.minimize(losses['g_loss'], var_list=g_vars)
d_train_op = d_optimizer.minimize(losses['d_loss'], var_list=d_vars)

saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

summary_op = tf.summary.merge_all()
input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 3])
input_mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 1])

output,l1_test = model.evaluate(input_image_tf, input_mask_tf, config=config, reuse=True)

output = (output + 1) * 127.5
output = tf.minimum(tf.maximum(output[:, :, :, ::-1], 0), 255)
output = tf.cast(output, tf.uint8)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    if config.load_model_dir != '':
        print('[-] Loading the pretrained model from: {}'.format(config.load_model_dir))
        ckpt = tf.train.get_checkpoint_state(config.load_model_dir)
        if ckpt:
            # saver.restore(sess, tf.train.latest_checkpoint(config.load_model_dir))
            assign_ops = list(
                map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(config.load_model_dir, x.name)),
                    g_vars))
            sess.run(assign_ops)
            print("[*] Loading SUCCESS.")
        else:
            print("[x] Loading ERROR.")

    summary_writer = tf.summary.FileWriter(config.tensorboard_folder, sess.graph, flush_secs=30)

    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)
    file_l1_train = open(config.tensorboard_folder+"l1_train.txt", mode='w')
    file_g_loss_train = open(config.tensorboard_folder+"g_loss_train.txt", mode='w')    
    file_d_loss_train = open(config.tensorboard_folder+"d_loss_train.txt", mode='w')    
    file_d_loss_nogp_train = open(config.tensorboard_folder+"d_loss_nogp_train.txt", mode='w')    
    file_l1_test = open(config.tensorboard_folder+"l1_test.txt", mode='w')    
    for step in range(1, config.max_iters+1):

        if config.pretrain_network is False:
            for _ in range(5):
                _, d_loss = sess.run([d_train_op, losses['d_loss']])

        _, g_loss = sess.run([g_train_op, losses['g_loss']])
        if step % config.viz_steps == 0:
            l1_mean=0
            l1_mean_test=0
            for i in range(test_num+1):
                if config.mask_type == 'rect':
                    mask = generate_mask_rect(config.img_shapes, config.mask_shapes, 0)
                if i==1000:
                    image = cv2.imread("/data4/lijunjie/tensorbord/test_img/Places365_val_00016284.png")
                    mask = (cv2.imread("/data4/lijunjie/tensorbord/test_img/Places365_val_00016284_mask.png")/255). astype(np.float32)[:,:,0:1]
                else:
                
                    image = cv2.imread(pathfile[i])
                    index=pathfile[i].rfind('/')
                    name=pathfile[i][index+1:len(pathfile[i])]
                h, w = image.shape[:2]
                image_ori=image
                if h >= config.img_shapes[0] and w >= config.img_shapes[1]:
                    h_start = (h-config.img_shapes[0]) // 2
                    w_start = (w-config.img_shapes[1]) // 2
                    image = image[h_start: h_start+config.img_shapes[0], w_start: w_start+config.img_shapes[1], :]
                else:
                    t = min(h, w)
                    image = image[(h-t)//2:(h-t)//2+t, (w-t)//2:(w-t)//2+t, :]
                    image = cv2.resize(image, (config.img_shapes[1], config.img_shapes[0]))

        # cv2.imwrite(os.path.join(config.saving_path, 'gt_{:03d}.png'.format(i)), image.astype(np.uint8))
                image = image * (1-mask) + 255 * mask
                #if i==0:
                    #cv2.imwrite(os.path.join(config.tensorboard_folder, '0input_'+name), image.astype(np.uint8))
                    #cv2.imwrite(os.path.join(config.tensorboard_folder, name), result[0][:, :, ::-1])
                assert image.shape[:2] == mask.shape[:2]

                h, w = image.shape[:2]
                grid = 4
                image = image[:h // grid * grid, :w // grid * grid, :]
                mask = mask[:h // grid * grid, :w // grid * grid, :]

                image = np.expand_dims(image, 0)
                mask = np.expand_dims(mask, 0)

                result,l1_test_loss = sess.run([output,l1_test], feed_dict={input_image_tf: image, input_mask_tf: mask})
                result_center=(result[0,64:192,64:192,::-1].astype(np.float32))/127.5-1
                image_center=(image_ori[64:192,64:192,:].astype(np.float32))/127.5-1                
                if i==1000:
                    #cv2.imwrite(os.path.join(config.tensorboard_folder, '0input_'+name), image.astype(np.uint8))
                    cv2.imwrite(os.path.join(config.tensorboard_folder, str(step)+'_'+name), result[0][:, :, ::-1])                
                l1_mean=l1_mean+l1_test_loss
                l1_mean_test=l1_mean_test+np.mean(np.abs(result_center-image_center))
            #file_l1_test.write('pppppppppppp')
            file_l1_test.write(str(l1_mean_test/1000))            
            #file_l1_test.close()
            #exit(0)
            print(str(l1_mean/1000),str(l1_mean_test/1000))
            file_l1_test.write('\n')            
        if step % config.viz_steps == 0:
            print('[{:04d}, {:04d}] G_loss > {}'.format(step // config.train_spe, step % config.train_spe, g_loss))
            summary_writer.add_summary(sess.run(summary_op), global_step=step)
            l1_loss_only, g_loss,d_loss,d_loss_nogp = sess.run([losses['l1_loss_only'], losses['g_loss'],losses['d_loss'],losses['d_loss_nogp']])
            file_l1_train.write(str(l1_loss_only))
            file_l1_train.write('\n')
            file_g_loss_train.write(str(g_loss))
            file_g_loss_train.write('\n')
            file_d_loss_train.write(str(d_loss))
            file_d_loss_train.write('\n')
            file_d_loss_nogp_train.write(str(d_loss_nogp))
            file_d_loss_nogp_train.write('\n')
        if step % (config.train_spe) == 0:
            saver.save(sess, os.path.join(config.tensorboard_folder, config.model_prefix), step)
    file_l1_train.close()
    file_g_loss_train.close()
    file_d_loss_train.close()
    file_d_loss_nogp_train.close()
    file_l1_test.close()

    coord.request_stop()
    coord.join(thread)
