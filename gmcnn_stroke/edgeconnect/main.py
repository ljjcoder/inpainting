import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
from shutil import copyfile
from src.config import Config
from src.edge_connect import EdgeConnect

class edge_module(nn.Module):
    def __init__(self):
        super(edge_module, self).__init__()
        config = self.load_config(intput_image_path='', mask_path='', mode=2)
        self.model = EdgeConnect(config)
        self.model.load()
        
    
    def inference(self, intput_image_path, mask_path, mode=2):
        return self.main(intput_image_path, mask_path, mode=2)

        
    def main(self, intput_image_path, mask_path, mode=None):
        r"""starts the model

        Args:
            mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
        """

        config = self.load_config(intput_image_path, mask_path, mode)


        # cuda visble devices
        # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


        # init device
        if torch.cuda.is_available():
            config.DEVICE = torch.device("cuda")
            torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
        else:
            config.DEVICE = torch.device("cpu")



        # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
        cv2.setNumThreads(0)


        # initialize random seed
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)



        # build the model and initialize
        # model = EdgeConnect(config)
        # model.load()


        # model training
        if config.MODE == 1:
            config.print()
            print('\nstart training...\n')
            model.train()

        # model test
        elif config.MODE == 2:
            print('\nstart testing...\n')
            output_results = self.model.test(config)
            return output_results

        # eval mode
        else:
            print('\nstart eval...\n')
            model.eval()


    def load_config(self, intput_image_path, mask_path, mode=None):
        r"""loads model config

        Args:
            mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
        """

        # parser = argparse.ArgumentParser()
        # parser.add_argument('--load_model_dir', type=str, default='')
        # parser.add_argument('--load_model_dir1', type=str, default='')
        # parser.add_argument('--img_shapes', type=str, default='')
        # parser.add_argument('--mode', type=str, default='')
        
        # parser.add_argument('--path', '--checkpoints', type=str, default='./checkpoints/places2_authormodel', help='model checkpoints path (default: ./checkpoints)')
        # parser.add_argument('--model', type=int, default=4, choices=[1, 2, 3, 4], help='1: edge model, 2: inpaint model, 3: edge-inpaint model, 4: joint model')
        # parser.add_argument('--input', type=str,default=intput_image_path, help='path to the input images directory or an input image')
        # parser.add_argument('--mask', type=str, default=mask_path, help='path to the masks directory or a mask file')
        # # parser.add_argument('--edge', type=str, help='path to the edges directory or an edge file')
        # parser.add_argument('--output', type=str, default='edgeconnect_result', help='path to the output directory')

        # parser.add_argument('opts',
        #                     help='all options',
        #                     default=None,
        #                     nargs=argparse.REMAINDER
        # )

        # test mode
        # if mode == 2:
        
        path = './edgeconnect/checkpoints/places2_authormodel'
        ec_model = 4
        ec_input = intput_image_path
        ec_mask = mask_path
        ec_output = './edgeconnect_result'
        

        # args = parser.parse_args()
        
        config_path = os.path.join(path, 'config.yml')

        # create checkpoints path if does't exist
        if not os.path.exists(path):
            os.makedirs(path)

        # copy config template if does't exist
        if not os.path.exists(config_path):
            copyfile('./config.yml.example', config_path)

        # load config file
        config = Config(config_path)

        # train mode
        if mode == 1:
            config.MODE = 1
            if ec_model:
                config.MODEL = ec_model

        # test mode
        elif mode == 2:
            config.MODE = 2
            config.MODEL = ec_model if ec_model is not None else 3
            config.INPUT_SIZE = 0

            if ec_input is not None:
                config.TEST_FLIST = ec_input

            if ec_mask is not None:
                config.TEST_MASK_FLIST = ec_mask

            # if args.edge is not None:
            #     config.TEST_EDGE_FLIST = args.edge

            if ec_output is not None:
                config.RESULTS = ec_output

        # eval mode
        elif mode == 3:
            config.MODE = 3
            config.MODEL = ec_model if ec_model is not None else 3

        return config


if __name__ == "__main__":
    main()
