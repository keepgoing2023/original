import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils import data
import torch.multiprocessing as mp
from torchvision import transforms
import transforms as trans
import time
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from dataset import get_loader
import math
from Models.USOD_Net import ImageDepthNet
import os
import pytorch_iou
import pytorch_ssim
import utils
import Testing
from Evaluation import imain
import shutil


criterion = nn.BCEWithLogitsLoss()
ssim_loss = pytorch_ssim.SSIM(window_size=7, size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)
class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v
def save_loss(save_dir, whole_iter_num, epoch_total_loss, epoch_loss, epoch):
    fh = open(save_dir, 'a')
    epoch_total_loss = str(epoch_total_loss)
    epoch_loss = str(epoch_loss)
    fh.write('until_' + str(epoch) + '_run_iter_num' + str(whole_iter_num) + '\n')
    fh.write(str(epoch) + '_epoch_total_loss' + epoch_total_loss + '\n')
    fh.write(str(epoch) + '_epoch_loss' + epoch_loss + '\n')
    fh.write('\n')
    fh.close()


def adjust_learning_rate(optimizer, decay_rate=.1):
    update_lr_group = optimizer.param_groups
    for param_group in update_lr_group:
        print('before lr: ', param_group['lr'])
        param_group['lr'] = param_group['lr'] * decay_rate
        print('after lr: ', param_group['lr'])
    return optimizer


def save_lr(save_dir, optimizer):
    update_lr_group = optimizer.param_groups[0]
    fh = open(save_dir, 'a')
    fh.write('encode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('decode:update:lr' + str(update_lr_group['lr']) + '\n')
    fh.write('\n')
    fh.close()


def train_net(num_gpus, args):
    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args))


### bce_ssim_loss
def bce_ssim_loss(pred, target):
    bce_out = criterion(pred, target)
    ssim_out = 1 - ssim_loss(pred, target)
    loss = bce_out + ssim_out
    return loss


### bce_iou_loss
def bce_iou_loss(pred, target):
    bce_out = criterion(pred, target)
    iou_out = iou_loss(pred, target)
    loss = bce_out + iou_out
    return loss

### dice_loss
def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss
def eval(args, net, epoch):
    cudnn.benchmark = True

    net.cuda()
    net.eval()

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
            # d1, d2, d3, d4, d5, db, ud2, ud3, ud4, ud5, udb = outputs_saliency
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
        print('eval epoch:{}, cost:{}'.format(epoch, np.mean(time_list) * 1000))

def main(local_rank, num_gpus, args):
    save_path = args.save_path
    log, writer = utils.set_save_path(save_path, remove=False)
    cudnn.benchmark = True
    net = ImageDepthNet(args)
    net.train()
    net.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    base_params = [params for name, params in net.named_parameters() if ("backbone" in name)]
    other_params = [params for name, params in net.named_parameters() if ("backbone" not in name)]
    print('model: #params={}'.format(utils.compute_num_params(net, text=True)))
    optimizer = optim.Adam([{'params': base_params, 'lr': args.lr * 0.1},
                            {'params': other_params, 'lr': args.lr}])  # 对不同的网络参数使用不同的学习率具体的判定条件是什么？
    train_dataset = get_loader(args.trainset, args.data_root, args.img_size, mode='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=6, pin_memory=True, shuffle=True,
                              drop_last=True)

    print('''
        Starting training:
            Train steps: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
        '''.format(args.train_steps, args.batch_size, args.lr, len(train_loader.dataset)))

    N_train = len(train_loader) * args.batch_size

    #loss_weights = [1, 0.8, 0.8, 0.5, 0.5, 0.5]
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    criterion = nn.BCEWithLogitsLoss()
    whole_iter_num = 0
    iter_num = math.ceil(len(train_loader.dataset) / args.batch_size)
    max_val_1 = max_val_2 = max_val_3 = max_val_4 = max_val_6 = max_val_7 = -1e18
    max_val_5 = 1
    timer = Timer()
    # pbar = tqdm(total=args.train_steps, leave=False, desc='train')
    current_epoch = 1
    # args.resume = "/home/buchao/Project/KeepGoing/change_tc_usod/test_save/24.pth"
    # if args.resume:
    #     checkpoint = torch.load(args.resume)
    #     current_epoch = checkpoint["epoch"]
    #     net.load_state_dict(checkpoint['model'], strict=True)
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    for epoch in range(current_epoch, args.epochs + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, args.epochs)]
        print('Starting epoch {}/{}.'.format(epoch, args.epochs))
        print('epoch:{0}-------lr:{1}'.format(epoch, args.lr))

        epoch_total_loss = 0
        epoch_loss = 0

        for i, data_batch in enumerate(train_loader):
            if (i + 1) > iter_num: break

            images, depths, label_224, label_7, label_14, label_28, label_56, label_112, \
            contour_224, contour_7, contour_14, contour_28, contour_56, contour_112 = data_batch

            images, depths, label_224, contour_224 = Variable(images.cuda(local_rank, non_blocking=True)), \
                                                     Variable(depths.cuda(local_rank, non_blocking=True)), \
                                                     Variable(label_224.cuda(local_rank, non_blocking=True)), \
                                                     Variable(contour_224.cuda(local_rank, non_blocking=True))

            label_7, label_14, label_28, label_56, label_112 = Variable(label_7.cuda()), Variable(label_14.cuda()), Variable(label_28.cuda()), \
                                                      Variable(label_56.cuda()), Variable(label_112.cuda())  # label_7(b,1,7,7)  label_14(b,1,14,14)

            contour_14, contour_28, contour_56, contour_112 = Variable(contour_14.cuda()), \
                                                              Variable(contour_28.cuda()), \
                                                              Variable(contour_56.cuda()), Variable(contour_112.cuda())

            outputs_saliency = net(images, depths)

            d_224, d_112, d_56, d_28, d_14, d_7_c, db_7, ud_112, ud_56, ud_28, ud_14, ud_7_c, udb_7 = outputs_saliency
            # d1, d2, d3, d4, d5, db, ud2, ud3, ud4, ud5, udb = outputs_saliency

            bce_loss1 = criterion(d_224, label_224)
            bce_loss2 = criterion(d_112, label_112)
            bce_loss3 = criterion(d_56, label_56)
            bce_loss4 = criterion(d_28, label_28)
            bce_loss5 = criterion(d_14, label_14)
            bce_loss6 = criterion(d_7_c, label_7)
            # bce_loss7 = criterion(db_7, label_7)

            iou_loss1 = bce_iou_loss(d_224,  label_224)
            iou_loss2 = bce_iou_loss(ud_112, label_224)
            iou_loss3 = bce_iou_loss(ud_56, label_224)
            iou_loss4 = bce_iou_loss(ud_28, label_224)
            iou_loss5 = bce_iou_loss(ud_14, label_224)
            # iou_loss6 = bce_iou_loss(ud6, label_224)
            iou_loss7 = bce_iou_loss(ud_7_c, label_224)

            c_loss1 = bce_ssim_loss(d_224,  contour_224)
            c_loss2 = bce_ssim_loss(ud_112, label_224)
            c_loss3 = bce_ssim_loss(ud_56, label_224)
            c_loss4 = bce_ssim_loss(ud_28, label_224)
            c_loss5 = bce_ssim_loss(ud_14, label_224)
            # c_loss6 = bce_ssim_loss(ud6, label_224)
            c_loss7 = bce_ssim_loss(ud_7_c, label_224)

            d_loss1 = dice_loss(d_224,   label_224)
            d_loss2 = dice_loss(ud_112,  label_224)
            d_loss3 = dice_loss(ud_56,  label_224)
            d_loss4 = dice_loss(ud_28,  label_224)
            d_loss5 = dice_loss(ud_14,  label_224)
            # d_loss6 = dice_loss(ud6, label_224)
            d_loss7 = dice_loss(ud_7_c,  label_224)

            BCE_total_loss = bce_loss1 + bce_loss2 + bce_loss3 + bce_loss4 + bce_loss5 + bce_loss6
            IoU_total_loss = iou_loss1 + iou_loss2 + iou_loss3 + iou_loss4 + iou_loss5 + iou_loss7
            Edge_total_loss = c_loss1 + c_loss2 + c_loss3 + c_loss4 + c_loss5 + c_loss7
            Dice_total_loss = d_loss1 + d_loss2 + d_loss3 + d_loss4 + d_loss5 + d_loss7
            total_loss = Edge_total_loss + BCE_total_loss + IoU_total_loss + Dice_total_loss

            epoch_total_loss += total_loss.cpu().data.item()
            epoch_loss += bce_loss1.cpu().data.item()

            print(
                'whole_iter_num: {0} --- {1:.4f} --- total_loss: {2:.6f} --- bce loss: {3:.6f} --- e loss: {4:.6f}'.format(
                    (whole_iter_num + 1), (i + 1) * args.batch_size / N_train, total_loss.item(), bce_loss1.item(), c_loss1.item()
                    ))

            optimizer.zero_grad()

            total_loss.backward()

            optimizer.step()

            whole_iter_num += 1
            # pbar.update(1)

            if whole_iter_num == args.train_steps:
                torch.save(net.state_dict(),
                           args.save_model_dir + 'sam_bridge.pth')

            if whole_iter_num == args.stepvalue1 or whole_iter_num == args.stepvalue2:
                optimizer = adjust_learning_rate(optimizer, decay_rate=args.lr_decay_gamma)
                save_dir = './loss.txt'
                save_lr(save_dir, optimizer)
                print('have updated lr!!')

        # print('the iter_num is {}'.format(whole_iter_num))  # k
        if epoch % 2 == 0:
            torch.save({  # k
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # f 是一个用于格式化字符串的前缀，表示这个字符串是一个格式化字符串（formatted string）。在格式化字符串中，你可以使用大括号 {} 来插入变量或表达式的值。f 前缀表示在字符串中可以包含变量或表达式，这些变量或表达式会在字符串被创建时被替换为它们的值。
            }, os.path.join(args.save_path, f"{epoch}.pth"))  # k

        print('Epoch finished ! Loss: {}'.format(epoch_total_loss / iter_num))
        log_info.append('epoch_total_loss is {} '.format(epoch_total_loss / iter_num))
        log_info.append('epoch_loss is {} '.format(epoch_loss / iter_num))
        writer.add_scalars('epoch_loss', {'train G': epoch_loss / iter_num}, epoch)
        print('开始生成预测图')
        eval(args, net, epoch)
        mae, max_f, mean_f, max_e, mean_e, smeasure, avg_p, auc = imain.evaluate(args)
        log_info.append('val: mae ={:.4f}'.format(mae))
        writer.add_scalars('mae', {'val': mae}, epoch)
        log_info.append('val: max_f={:.4f}'.format(max_f))
        writer.add_scalars('max_f', {'val': max_f}, epoch)
        log_info.append('val: mean_f={:.4f}'.format(mean_f))
        writer.add_scalars('mean_f', {'val': mean_f}, epoch)
        log_info.append('val: max_e={:.4f}'.format(max_e))
        writer.add_scalars('max_e', {'val': max_e}, epoch)
        log_info.append('val: mean_e={:.4f}'.format(mean_e))
        writer.add_scalars('mean_e', {'val': mean_e}, epoch)
        log_info.append('val: smeasure={:.4f}'.format(smeasure))
        writer.add_scalars('smeasure', {'val': smeasure}, epoch)
        print(
            'epoch {}: {:.4f} mae || {:.4f} max-fm || {:.4f} mean-fm || {:.4f} max-Emeasure || {:.4f} mean-Emeasure || {:.4f} S-measure || {:.4f} AP || {:.4f} AUC.\n'
            .format(epoch, mae, max_f, mean_f, max_e,
                    mean_e, smeasure, avg_p, auc))
        if max_f > max_val_1 and mean_f > max_val_2 and mean_e > max_val_3 and smeasure > max_val_4 and mae < max_val_5:
            max_val_1 = max_f
            max_val_2 = mean_f
            max_val_3 = mean_e
            max_val_4 = smeasure
            max_val_5 = mae
            torch.save({  # k
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                # f 是一个用于格式化字符串的前缀，表示这个字符串是一个格式化字符串（formatted string）。在格式化字符串中，你可以使用大括号 {} 来插入变量或表达式的值。f 前缀表示在字符串中可以包含变量或表达式，这些变量或表达式会在字符串被创建时被替换为它们的值。
            }, os.path.join(args.save_best_path, f"{epoch}.pth"))  # k  # kg
        save_lossdir = './loss.txt'
        save_loss(save_lossdir, whole_iter_num, epoch_total_loss / iter_num, epoch_loss / iter_num, epoch + 1)
        t = timer.t()
        prog = whole_iter_num / args.train_steps
        t_epoch = utils.time_text(t - t_epoch_start)  # 完成一个epoch的时间
        t_elapsed, t_all = time_text(t), time_text(t / prog)  # 目前所用时间以及预计全部完成的时间
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))
        print('{}time: {}/{}'.format(t_epoch, t_elapsed, t_all))
        log(', '.join(log_info))
        writer.flush()  # g 用于将数据写入可视化系统并确保其立即可见。

        try:
            shutil.rmtree(args.save_test_path_root + '/USOD10K/')
            print(f"成功删除目录 {args.save_test_path_root + '/USOD10K/'} 及其内容。")
        except OSError as e:
            print(f"删除目录 {args.save_test_path_root + '/USOD10K/'} 时发生错误: {e}")


