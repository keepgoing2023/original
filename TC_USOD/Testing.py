import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import get_loader
import transforms as trans
from torchvision import transforms
import time
from Models.USOD_Net import ImageDepthNet
from torch.utils import data
import numpy as np
import os
# import matplotlib.pyplot as plt
# from PIL import Image

def test_net(args):
    cudnn.benchmark = True
    net = ImageDepthNet(args)
    net.cuda()
    net.eval()
    model_path = "/home/buchao/Project/KeepGoing/change_tc_usod/test_save/70.pth"
    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict['model'])
    test_paths = args.test_paths.split('+')

    for test_dir_img in test_paths:

        test_dataset = get_loader(test_dir_img, args.data_root, args.img_size, mode='test')

        test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)

        time_list = []
        for i, data_batch in enumerate(test_loader):
            images, depths, image_w, image_h, image_path = data_batch
            images, depths = Variable(images.cuda()), Variable(depths.cuda())

            starts = time.time()
            outputs_saliency = net(images, depths)
            ends = time.time()
            time_use = ends - starts
            time_list.append(time_use)

            d_224, d_112, d_56, d_28, d_14, d_7_c, db_7, ud_112, ud_56, ud_28, ud_14, ud_7_c, udb_7 = outputs_saliency
            image_w, image_h = int(image_w[0]), int(image_h[0])
            transform = trans.Compose([
                transforms.ToPILImage(),
                trans.Scale((image_w, image_h))
            ])
            output_s = d_224.data.cpu().squeeze(0)

            output_s = transform(output_s)

            dataset = test_dir_img.split('/')[0]
            filename = image_path[0].split('/')[-1].split('.')[0]  # 从一个文件路径中提取文件名（包括扩展名），然后去掉扩展名

            # save saliency maps
            save_test_path = args.save_test_path_root + dataset + '/USOD10K/'
            if not os.path.exists(save_test_path):
                os.makedirs(save_test_path)
            output_s.save(os.path.join(save_test_path, filename + '.png'))





