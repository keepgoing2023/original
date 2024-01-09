import torch
# import numpy as np
# import numpy as np
#
# # 创建一个二维数组
# x = np.array([[1, 2],
#               [4, 5],
#               ])
#
# # 对数组进行二维傅里叶变换
# fft_result = np.fft.fft2(x)
#
# # 将频率域中心移动到数组中心
# fft_result_shifted = np.fft.fftshift(fft_result)
# f0shift = np.fft.ifftshift(fft_result_shifted)
# img_back1 = np.fft.ifft2(f0shift)
# img_back = np.abs(img_back1)
#
# print("原始数组:\n", x)
# print("\n二维傅里叶变换结果:\n", f0shift)
# print("\n经过fftshift后的结果:\n", img_back1)
print(torch.cuda.is_available())
