import os
import torch
import Training
import Testing
from Evaluation import imain
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # train
    parser.add_argument('--Training', default=False, type=bool, help='Training or not')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:33111', type=str, help='init_method')
    parser.add_argument('--data_root', default='', type=str, help='data path')
    parser.add_argument('--train_steps', default=60000, type=int, help='train_steps')
    parser.add_argument('--img_size', default=224, type=int, help='network input size')
    # parser.add_argument('--SG_former_pretrained_model',
    #                     default="../pretrained_model/SG-Former-B.pth",
    #                     type=str, help='load pretrained model')
    parser.add_argument('--biformer_pretrained_model',
                        default="../pretrained_model/biformer_small_best.pth",
                        type=str, help='load pretrained model')
    # parser.add_argument('--pretrained_model', default='/home/buchao/Project/KeepGoing/USOD10K/pretrained_model/80.7_T2T_ViT_t_14.pth.tar', type=str, help='load pretrained model')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=int, help='learning rate decay')
    parser.add_argument('--lr', default=1e-4, type=int, help='learning rate')
    parser.add_argument('--epochs', default=70, type=int, help='epochs')
    parser.add_argument('--batch_size', default=8, type=int, help='batch_size')
    parser.add_argument('--stepvalue1', default=60000, type=int, help='the step 1 for adjusting lr')
    parser.add_argument('--stepvalue2', default=60000, type=int, help='the step 2 for adjusting lr')
    parser.add_argument('--trainset', default="/home/yanchaorui/Project/dataset/USOD10k/TR/", type=str, help='Trainging set')
    parser.add_argument('--save_model_dir', default="../the last pth/", type=str, help='save model path')
    parser.add_argument('--save_path', default="../test_save/", type=str, help='save model path')
    parser.add_argument('--save_best_path', default="../test_save/save_best/", type=str, help='save best model path')
    # parser.add_argument('--resume', default="/home/buchao/Project/KeepGoing/change_tc_usod/save/58.pth", type=str, help='resume')
    # # test
    parser.add_argument('--Testing', default=False, type=bool, help='Testing or not')
    parser.add_argument('--save_test_path_root', default="../save salient maps/", type=str, help='save saliency maps path')  #k
    parser.add_argument('--test_paths', type=str, default="/home/yanchaorui/Project/dataset/USOD10k/TE/")  # k

    # evaluation
    parser.add_argument('--Evaluation', default=False, type=bool, help='Evaluation or not')
    parser.add_argument('--methods', type=str, default='USOD10K', help='evaluated method name')
    parser.add_argument('--save_dir', type=str, default="/home/buchao/Project/KeepGoing/change_tc_usod/heihei/", help='path for saving result.txt')
    # parser.add_argument('--pth_save_dir', type=str, default='/home/buchao/Project/KeepGoing/USOD10K/pinlv/all_pth', help='pth_save')  # k

    args = parser.parse_args()

    args.Training = 'True'

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # args.Testing = 'True'
    args.save_dir = "../save_result/"
    # args.Evaluation = 'True'

    num_gpus = torch.cuda.device_count()
    if args.Training:
        # Training.train_net(num_gpus=num_gpus, args=args)
        Training.main(0, num_gpus=num_gpus, args=args)
    if args.Testing:
        Testing.test_net(args)
    if args.Evaluation:
        imain.evaluate(args)