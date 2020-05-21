import numpy as np
import cv2
import os
def make_data(config,path_image,path_mask,saving_path,i):
    mask = (cv2.imread(path_mask)/255). astype(np.float32)[:,:,0:1]
             
    image = cv2.imread(path_image)
    index=path_image.rfind('/')
    name=path_image[index+1:len(path_image)]

    h, w = image.shape[:2]

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
    if i==0:
        cv2.imwrite(os.path.join(saving_path, '0input_'+name[:len(name)-3]+'bmp'), image.astype(np.uint8))
    #print(image.shape,mask.shape)
    #exit(0)
    assert image.shape[:2] == mask.shape[:2]

    h, w = image.shape[:2]
    grid = 4
    image = image[:h // grid * grid, :w // grid * grid, :]
    mask = mask[:h // grid * grid, :w // grid * grid, :]

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    return image, mask,name