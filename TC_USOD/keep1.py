import matplotlib.pyplot as plt
import torch
import os
from PIL import Image
import numpy as np
from dataset import get_loader
from torch.utils import data
test_dir_img = "/home/buchao/Project/KeepGoing/dataset/USOD10k/tte/"
data_root = ""
img_size = 224
test_dataset = get_loader(test_dir_img, data_root, img_size, mode='test')
test_loader = data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)


def fft(x, rate, prompt_type):
    mask = np.zeros(x.shape)
    w, h = x.shape[-2:]
    line = int((w * h * rate) ** 0.5 // 2)
    mask[:, :, w // 2 - line:w // 2 + line, h // 2 - line:h // 2 + line] = 1

    fft = np.fft.fftshift(np.fft.fft2(x))

    if prompt_type == 'highpass':
        fft = fft * (1 - mask)
    elif prompt_type == 'lowpass':
        fft = fft * mask
    fr = fft.real
    fi = fft.imag

    fft_hires = np.fft.ifftshift(fr + 1j * fi)
    inv = np.fft.ifft2(fft_hires).real

    inv = np.abs(inv)
    return inv
# for i, data_batch in enumerate(test_loader):
#     images, depths, image_w, image_h, image_path = data_batch
#     f_image = fft(images, 0.25, "highpass")
#     f_depth = fft(depths, 0.25, "highpass")
#     i_no_batch = np.squeeze(f_image, 0)
#     d_no_batch = np.squeeze(f_depth, 0)
#     image_pil = Image.fromarray(i_no_batch, 'RGB')
#     depth_pil = Image.fromarray(d_no_batch, 'RGB')
#     image_pil.show()
#     save_image_path = os.path.join("/home/buchao/Project/KeepGoing/dataset/USOD10k/image/", f"{i}_image.png")
#     save_depth_path = os.path.join("/home/buchao/Project/KeepGoing/dataset/USOD10k/depth/", f"{i}_depth.png")
#     image_pil.save(save_image_path)
#     depth_pil.save(save_depth_path)
#     print('第{}个'.format(i))
im = np.array(Image.open("/home/buchao/Project/KeepGoing/dataset/USOD10k/tte/RGB/00011.png"))
im = np.expand_dims(im, axis=0)
swapped_array = np.transpose(im, (0, 3, 1, 2))
f_image = fft(im, 1.0, "highpass")
i_no_batch = np.squeeze(f_image, 0)
image_pil = Image.fromarray(i_no_batch, 'RGB')
save_image_path = os.path.join("/home/buchao/Project/KeepGoing/dataset/USOD10k/image/", f"{6}_image.png")
image_pil.save(save_image_path)
print('测试')






