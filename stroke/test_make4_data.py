import numpy as np
import cv2
import os
import subprocess
import glob
import tensorflow as tf
from options.test_options import TestOptions
from util.util import generate_mask_rect, generate_mask_stroke
from net.network import GMCNNModel
import numpy as np
print(str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
        "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]
        )))
#exit(0)
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
        "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]
        ))

config = TestOptions().parse()

if os.path.isfile(config.dataset_path):
    pathfile = open(config.dataset_path, 'rt').read().splitlines()
elif os.path.isdir(config.dataset_path):
    pathfile = glob.glob(os.path.join(config.dataset_path, '*.png'))
else:
    print('Invalid testing data file/folder path.')
    exit(1)
total_number = len(pathfile)
test_num = total_number if config.test_num == -1 else min(total_number, config.test_num)
print('The total number of testing images is {}, and we take {} for test.'.format(total_number, test_num))

model = GMCNNModel()

reuse = False
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = False
with tf.Session(config=sess_config) as sess:
    input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 3])
    input_mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 1])

    output = model.evaluate(input_image_tf, input_mask_tf, config=config, reuse=reuse)
    output = (output + 1) * 127.5
    output = tf.minimum(tf.maximum(output[:, :, :, ::-1], 0), 255)
    output = tf.cast(output, tf.uint8)

    # load pretrained model
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    assign_ops = list(map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(config.load_model_dir, x.name)),
                          vars_list))
    sess.run(assign_ops)
    print('Model loaded.')
    total_time = 0

    if config.random_mask:
        np.random.seed(config.seed)
  
    for root, dirs, _ in os.walk('/data4/lijunjie/mini-imagenet-tools/processed_images/train'):
        #for f in files:
            #print(os.path.join(root, f))

        for d in dirs:
            path=os.path.join(root, d)
            path_1=path.replace('train','train_gmcnn_1')
            path_2=path.replace('train','train_gmcnn_2')
            path_3=path.replace('train','train_gmcnn_3')
            path_4=path.replace('train','train_gmcnn_4')
            path_mask_1=path.replace('train','mask/train_1')
            path_mask_2=path.replace('train','mask/train_2')
            path_mask_3=path.replace('train','mask/train_3')
            path_mask_4=path.replace('train','mask/train_4')            
            if not os.path.isdir(path_1):            
                os.mkdir(path_1)
                os.mkdir(path_2)
                os.mkdir(path_3)
                os.mkdir(path_4)
            files = os.listdir(path) 
            #images=[]
            #imgs_gray=[]
            #Xt_img_ori=[]
            Paths=[]
            train_dir=[]
            print(path_1)
            Paths.append(path_1)
            Paths.append(path_2)
            Paths.append(path_3)
            Paths.append(path_4)
            train_dir.append(path_mask_1)
            train_dir.append(path_mask_2) 
            train_dir.append(path_mask_3)
            train_dir.append(path_mask_4)             
            for file in files:        
        # cv2.imwrite(os.path.join(config.saving_path, 'gt_{:03d}.png'.format(i)), image.astype(np.uint8))
                print(file)
                print(os.path.join(path, file))
                #exit(0)
                for maskid in range(4):
                    image = cv2.imread(os.path.join(path, file))
                    #index=pathfile[i].rfind('/')
                    name=file
                    mask=(cv2.imread(os.path.join(train_dir[maskid], file))/255). astype(np.float32)[:,:,0:1] 

                    h, w = image.shape[:2]

                    if h >= config.img_shapes[0] and w >= config.img_shapes[1]:
                        h_start = (h-config.img_shapes[0]) // 2
                        w_start = (w-config.img_shapes[1]) // 2
                        image = image[h_start: h_start+config.img_shapes[0], w_start: w_start+config.img_shapes[1], :]
                    else:
                        t = min(h, w)
                        image = image[(h-t)//2:(h-t)//2+t, (w-t)//2:(w-t)//2+t, :]
                        image = cv2.resize(image, (config.img_shapes[1], config.img_shapes[0]))        
                        image = image * (1-mask) + 255 * mask
        #cv2.imwrite("/home/lijunjie/generative_inpainting-master_global_local/data/random_rec_places_10000/"+name,(mask[:,:,0]*255).astype(dtype=np.uint8))        
                    #if i==0:
                       # cv2.imwrite(os.path.join(config.save_root2, file), image.astype(np.uint8))
        #print(image.shape,mask.shape)
        #exit(0)
                    assert image.shape[:2] == mask.shape[:2]

                    h, w = image.shape[:2]
                    grid = 4
                    image = image[:h // grid * grid, :w // grid * grid, :]
                    mask = mask[:h // grid * grid, :w // grid * grid, :]

                    image = np.expand_dims(image, 0)
                    mask = np.expand_dims(mask, 0)

                    result = sess.run(output, feed_dict={input_image_tf: image, input_mask_tf: mask})
                    cv2.imwrite(Paths[maskid]+'/'+file, result[0][:, :, ::-1])
                    #exit()
        #cv2.imwrite("/home/lijunjie/generative_inpainting-master_global_local/data/random_rec_places_10000/"+name,(mask[:,:,0]*255).astype(dtype=np.uint8))
            #print(' > {} / {}'.format(i+1, test_num))
print('done.')
