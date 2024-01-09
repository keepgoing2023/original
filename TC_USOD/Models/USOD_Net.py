from .Transformer_depth import Transformer
from .Transformer import token_Transformer
from .DAM_module import *
from .Decoder_Dconv import Decoder

from .BaseBlock import *
# from .cmWR import *
# from .sgformer import *
from .biformer import *
from .SAM_decoder import  *
from .Adaptive import *

class ImageDepthNet(nn.Module):
    def __init__(self, args):
        super(ImageDepthNet, self).__init__()
        # Cross modality fusion
        self.DAM2 = CA_SA_Enhance(128)
        self.DAM3 = CA_SA_Enhance(256)
        self.DAM4 = CA_SA_Enhance(512)
        self.DAM5 = CA_SA_Enhance(1024)

        # BiFormer Encoder
        self.rgb_backbone = biformer_small(pretrained=True, args=args)
        self.depth_backbone = biformer_small(pretrained=True, args=args)
        # Encoder
        # self.rgb_backbone = T2t_vit_t_14(pretrained=True, args=args)
        # self.depth_backbone = T2t_vit_t_14(pretrained=True, args=args)
        # self.rrgb_backbone = T2t_vit_t_14(pretrained=True, args=args)  # k

        # SG_former encoder
        # self.rgb_backbone = sgformer_b(pretrained=True, args=args)
        # self.depth_backbone = sgformer_b(pretrained=True, args=args)
        # self.rrgb_backbone = sgformer_b(pretrained=True, args=args)

        # self.transformer_16 = Transformer(embed_dim=512, depth=12, num_heads=8, mlp_ratio=3.)
        # self.transformer_32 = Transformer(embed_dim=768, depth=12, num_heads=6, mlp_ratio=3.)
        # self.rgbd_transformer = Transformer(embed_dim=384, depth=12, num_heads=6, mlp_ratio=3.)  # k

        # 自适应
        self.adaptive_1 = Adaptive(embedding_dim=64, num_heads=8, mlp_dim=128)
        self.adaptive_2 = Adaptive(embedding_dim=128, num_heads=8, mlp_dim=256)
        self.adaptive_3 = Adaptive(embedding_dim=256, num_heads=8, mlp_dim=512)
        self.adaptive_4 = Adaptive(embedding_dim=512, num_heads=8, mlp_dim=1024)
        # SAM decoder
        # self.sam_decoder = TwoWayTransformer(depth=2, embedding_dim=384, num_heads=8, mlp_dim=98)

        # # self-modality attention refinement
        # self.ca_rgb_5 = ChannelAttention(768)
        # self.ca_rgb_4 = ChannelAttention(384)
        # self.ca_rgb_3 = ChannelAttention(192)
        # self.ca_rgb_2 = ChannelAttention(96)
        #
        # self.sa_rgb_5 = SA_Enhance(kernel_size=7)
        # self.sa_rgb_4 = SA_Enhance(kernel_size=7)
        # self.sa_rgb_3 = SA_Enhance(kernel_size=7)
        # self.sa_rgb_2 = SA_Enhance(kernel_size=7)
        # self.sa_depth = SA_Enhance(kernel_size=7)
        # self.sa_rrgb = SA_Enhance(kernel_size=7)
        #
        # # cross-modality weighting refinement
        # self.conv_rgb = BaseConv2d(384, 384, kernel_size=3, padding=1)
        # self.conv_depth = BaseConv2d(384, 384, kernel_size=3, padding=1)
        # self.conv_rrgbd = BaseConv2d(384, 384, kernel_size=3, padding=1)
        # self.cmWR = cmWR(384, squeeze_ratio=1)

        # Decoder
        self.decoder = Decoder()

    def forward(self, image_Input, depth_Input):  # image_Input:(8,3,224,224)  depth_Input:(8,3,224,224)
        B, _, _, _ = image_Input.shape

        # feature_map1 = self.rgb_backbone(image_Input, layer_flag=1)  # feature_map1:(b,3,244,244)
        feature_map1 = self.rgb_backbone(image_Input, layer_flag=1)
        # feature_map1, feature_map2, feature_map3, feature_map4, feature_map5
        dep_map1 = self.depth_backbone(depth_Input, layer_flag=1)
        # dep_layer1:(8,3,224,224) dep_layer2:(8,64,56 ,56) dep_layer3:(8,64,28,28) dep_layer_vit:(8,196,384)
        # rgb_map1:(b,3,224,224) rgb_map2:(b,96,56,56) rgb_map3:(b,192,28,28) rgb_map4:(b,384,14,14) rgb_map5:(b,768,7,7) rgb_1_4:(b,3136,96) rgb_1_8(b,784,192) rgb_1_16(8,196,384) rgb_1_32(8,49,768)

        feature_map2 = self.rgb_backbone(feature_map1, layer_flag=2)  # feature_map2:(8,64,56,56) rgb_fea_1_4:(8,3136,96)
        dep_map2 = self.depth_backbone(dep_map1, layer_flag=2)
        # print("feature_map2 shape is {}".format(feature_map2.shape))
        # print(11)
        img_cmf2 = self.DAM2(feature_map2, dep_map2)  # img_cmf2:(8,64,56,56)  biformer_base(b,96,56,56)
        cha_CAM_fea_1, cha_img_fea_1 = self.adaptive_1(img_cmf2, feature_map2)
        cha_img_fea_1 = cha_img_fea_1 + feature_map2
        cha_CAM_fea_1 = cha_CAM_fea_1 + dep_map2
        # img_layer_cat2 = feature_map2 + img_cmf2  # img_layer_cat2:(8,64,56,56)

        feature_map3 = self.rgb_backbone(cha_img_fea_1, layer_flag=3)  # feature_map3:(8,128,28,28)
        dep_map3 = self.depth_backbone(cha_CAM_fea_1, layer_flag=3)
        img_cmf3 = self.DAM3(feature_map3, dep_map3)  # img_cmf3:(8,128,28,28) biformer_base(b,192,28,28)
        # img_layer_cat3 = feature_map3 + img_cmf3  # img_layer_cat3:(8,128,28,28)
        cha_CAM_fea_2, cha_img_fea_2 = self.adaptive_2(img_cmf3, feature_map3)
        cha_img_fea_2 = cha_img_fea_2 + feature_map3
        cha_CAM_fea_2 = cha_CAM_fea_2 + dep_map3

        feature_map4 = self.rgb_backbone(cha_img_fea_2, layer_flag=4)  # feature_map4:(b,256,14,14)
        dep_map4 = self.depth_backbone(cha_CAM_fea_2, layer_flag=4)
        img_cmf4 = self.DAM4(feature_map4, dep_map4)  # img_cmf3:(8,256,14,14) biformer_base(b,384,14,14)
        # img_layer_cat4 = feature_map4 + img_cmf4  # img_layer_cat3:(8,256,14,14)
        cha_CAM_fea_3, cha_img_fea_3 = self.adaptive_3(img_cmf4, feature_map4)
        cha_img_fea_3 = cha_img_fea_3 + feature_map4
        cha_CAM_fea_3 = cha_CAM_fea_3 + dep_map4


        feature_map5 = self.rgb_backbone(cha_img_fea_3, layer_flag=5)  # feature_map5:(8,512,7,7)
        dep_map5 = self.depth_backbone(cha_CAM_fea_3, layer_flag=5)
        img_cmf5 = self.DAM5(feature_map5, dep_map5)  # biformer_base(b,768,7,7)
        cha_CAM_fea_4, cha_img_fea_4 = self.adaptive_4(img_cmf5, feature_map5)


        # dep_vit = dep_map5.flatten(2).transpose(1, 2)
        # img_vit = feature_map5.flatten(2).transpose(1, 2)
        # rgb_fea_1_16, depth_fea_1_16 = self.transformer_16(img_vit,
        #                                                    dep_vit)  # rgb_fea_1_16(b,49,512) depth_fea_1_16(b,49,512)
        #
        # rgb_fea_1_16 = rgb_fea_1_16.transpose(1, 2).reshape(B, 512, 7, 7)
        # depth_fea_1_16 = depth_fea_1_16.transpose(1, 2).reshape(B, 512, 7, 7)

        # 初步想的是将depth作为sam中的prompt
        # 第一次是 k = rgb_fea_1_16, q = depth_fea_1_16
        # rgb_fea, dep_fea = self.sam_decoder(rgb_fea_1_16, depth_fea_1_16)
        # rgb_fea = rgb_fea.transpose(1, 2).reshape(B, 384, 14, 14)
        # dep_fea = dep_fea.transpose(1, 2).reshape(B, 384, 14, 14)

        outputs = self.decoder.forward(cha_img_fea_4, cha_CAM_fea_4, feature_map4, feature_map3, feature_map2, feature_map1)

        return outputs

# 更换backbone代码,增加cwAR模块。
#     def forward(self, image_Input, depth_Input):  # image_Input:(8,3,224,224)  depth_Input:(8,3,224,224)
#         B, _, _, _ = image_Input.shape
#         # image_test = self.rgb_backbone(image_Input)  # image_test(b,1000)
#
#         feature_map1 = self.rgb_backbone(image_Input, layer_flag=1)  # feature_map1:(b,3,244,244)
#
#         # dep_layer3_vit, _, _, dep_layer1, dep_layer2, dep_layer3 = self.depth_backbone(depth_Input)
#         # dep_layer1:(8,3,224,224) dep_layer2:(8,64,56 ,56) dep_layer3:(8,64,28,28) dep_layer_vit:(8,196,384)
#
#         dep_map1, dep_map2, dep_map3, dep_map4, dep_map5, dep_1_4, dep_1_8, dep_1_16, dep_1_32 = self.depth_backbone(depth_Input)
#         rgb_map1, rgb_map2, rgb_map3, rgb_map4, rgb_map5, rgb_1_4, rgb_1_8, rgb_1_16, rgb_1_32 = self.rrgb_backbone(image_Input)  # 对rgb图像处理。
#         # rgb_map1:(b,3,224,224) rgb_map2:(b,96,56,56) rgb_map3:(b,192,28,28) rgb_map4:(b,384,14,14) rgb_map5:(b,768,7,7) rgb_1_4:(b,3136,96) rgb_1_8(b,784,192) rgb_1_16(8,196,384) rgb_1_32(8,49,768)
#
#         # rgb自注意力
#         rgb_map5_SA = self.sa_rgb_5(rgb_map5).view(B, -1, 49)  # rgb_map5_SA:(b,1,7,7)
#         rgb_map4_SA = self.sa_rgb_4(rgb_map4).view(B, -1, 196)  # rgb_map4_SA:(b,1,14,14)
#         rgb_map3_SA = self.sa_rgb_3(rgb_map3).view(B, -1, 784)  # rgb_map3_SA:(b,1,28,28)
#         rgb_map2_SA = self.sa_rgb_2(rgb_map2).view(B, -1, 3136)  # rgb_map3_SA:(b,1,56,56)
#
#         rgb_map5_CA = self.ca_rgb_5(rgb_map5).view(B, 768, -1)  # rgb_map5_SA:(b,768,1,1)
#         rgb_map4_CA = self.ca_rgb_4(rgb_map4).view(B, 384, -1)  # rgb_map4_SA:(b,384,1,1)
#         rgb_map3_CA = self.ca_rgb_3(rgb_map3).view(B, 192, -1)  # rgb_map3_SA:(b,192,1,1)
#         rgb_map2_CA = self.ca_rgb_2(rgb_map2).view(B, 96, -1)  # rgb_map3_SA:(b,96,1,1)
#
#         rgb_map5_W = torch.bmm(rgb_map5_CA, rgb_map5_SA).view(B, 768, 7, 7)
#         rgb_map4_W = torch.bmm(rgb_map4_CA, rgb_map4_SA).view(B, 384, 14, 14)
#         rgb_map3_W = torch.bmm(rgb_map3_CA, rgb_map3_SA).view(B, 192, 28, 28)
#         rgb_map2_W = torch.bmm(rgb_map2_CA, rgb_map2_SA).view(B, 96, 56, 56)
#
#         rgb_map5_CS = rgb_map5 * rgb_map5_W + rgb_map5
#         rgb_map4_CS = rgb_map4 * rgb_map4_W + rgb_map4
#         rgb_map3_CS = rgb_map3 * rgb_map3_W + rgb_map3
#         rgb_map2_CS = rgb_map2 * rgb_map2_W + rgb_map2
#
#         # depth自注意力
#         dep_map5_SA = self.sa_rgb_5(dep_map5).view(B, -1, 49)  # dep_map5_SA:(b,1,7,7)
#         dep_map4_SA = self.sa_rgb_4(dep_map4).view(B, -1, 196)  # dep_map4_SA:(b,1,14,14)
#         dep_map3_SA = self.sa_rgb_3(dep_map3).view(B, -1, 784)  # dep_map3_SA:(b,1,28,28)
#         dep_map2_SA = self.sa_rgb_2(dep_map2).view(B, -1, 3136)  # dep_map3_SA:(b,1,56,56)
#
#         dep_map5_CA = self.ca_rgb_5(dep_map5).view(B, 768, -1)  # dep_map5_SA:(b,768,1,1)
#         dep_map4_CA = self.ca_rgb_4(dep_map4).view(B, 384, -1)  # dep_map4_SA:(b,384,1,1)
#         dep_map3_CA = self.ca_rgb_3(dep_map3).view(B, 192, -1)  # dep_map3_SA:(b,192,1,1)
#         dep_map2_CA = self.ca_rgb_2(dep_map2).view(B, 96, -1)  # dep_map3_SA:(b,96,1,1)
#
#         dep_map5_W = torch.bmm(dep_map5_CA, dep_map5_SA).view(B, 768, 7, 7)
#         dep_map4_W = torch.bmm(dep_map4_CA, dep_map4_SA).view(B, 384, 14, 14)
#         dep_map3_W = torch.bmm(dep_map3_CA, dep_map3_SA).view(B, 192, 28, 28)
#         dep_map2_W = torch.bmm(dep_map2_CA, dep_map2_SA).view(B, 96, 56, 56)
#
#         dep_map5_CS = dep_map5 * dep_map5_W + dep_map5
#         dep_map4_CS = dep_map4 * dep_map4_W + dep_map4
#         dep_map3_CS = dep_map3 * dep_map3_W + dep_map3
#         dep_map2_CS = dep_map2 * dep_map2_W + dep_map2
#         # print("rgb_map5_SA shape is {}".format(rgb_map5_SA.shape))
#
#         img_cmf1 = self.DAM1(rgb_map1, dep_map1)  # imf_cf1:(8,3,224,224)
#         img_layer_cat1 = feature_map1 + img_cmf1  # img_layer_cat1:(8,3,224,224)
#         feature_map2, rgb_fea_1_4 = self.rgb_backbone(img_layer_cat1, layer_flag=2)  # feature_map2:(8,96,56,56) rgb_fea_1_4:(8,3136,96)
#         # print("feature_map2 shape is {}".format(feature_map2.shape))
#         # print(11)
#
#         img_cmf2 = self.DAM2(rgb_map2_CS, dep_map2_CS)  # img_cmf2:(8,64,56,56)
#         img_layer_cat2 = feature_map2 + img_cmf2  # img_layer_cat2:(8,6 4,56,56)
#         feature_map3, rgb_fea_1_8 = self.rgb_backbone(img_layer_cat2, layer_flag=3)  # feature_map3:(8,64,28,28) rgb_fea_1_8:(8,784,64)
#
#         img_cmf3 = self.DAM3(rgb_map3_CS, dep_map3_CS)  # img_cmf3:(8,64,28,28)
#         img_layer_cat3 = feature_map3 + img_cmf3  # img_layer_cat3:(8,64,28,28)
#         feature_map4, rgb_fea_1_16 = self.rgb_backbone(img_layer_cat3, layer_flag=4)
#
#         img_cmf4 = self.DAM4(rgb_map4_CS, dep_map4_CS)  # img_cmf3:(8,64,28,28)
#         img_layer_cat4 = feature_map4 + img_cmf4  # img_layer_cat3:(8,64,28,28)
#         feature_map5, rgb_fea_1_32 = self.rgb_backbone(img_layer_cat4, layer_flag=5)
#
#         rgb_fea_1_16, depth_fea_1_16 = self.transformer_16(rgb_fea_1_16, dep_1_16)  # 更改后 rgb_fea_1_16(b,196,384) depth_fea_1_16(b,196,384)
#         rgb_fea_1_32, depth_fea_1_32 = self.transformer_32(rgb_fea_1_32, dep_1_32)  # 更改后 rgb_fea_1_32(b,49,768) depth_fea_1_16(b,49,768)
#
#         rgb_fea_1_16 = rgb_fea_1_16.transpose(1, 2).reshape(B, 384, 14, 14)
#         depth_fea_1_16 = depth_fea_1_16.transpose(1, 2).reshape(B, 384, 14, 14)
#         rgb_fea_1_32 = rgb_fea_1_32.transpose(1, 2).reshape(B, 768, 7, 7)
#         depth_fea_1_32 = depth_fea_1_32.transpose(1, 2).reshape(B, 768, 7, 7)
#
#         outputs = self.decoder.forward(rgb_fea_1_16, depth_fea_1_16, rgb_fea_1_32, depth_fea_1_32, feature_map5, feature_map4, feature_map3, feature_map2, feature_map1)
#
#
#         return outputs



#（3）去除cam
    # def forward(self, image_Input, depth_Input):  # image_Input:(8,3,224,224)  depth_Input:(8,3,224,224)
    #     B, _, _, _ = image_Input.shape
    #     feature_map1 = self.rgb_backbone(image_Input, layer_flag=1)  # feature_map1:(b,3,244,244)
    #
    #     dep_map1, dep_map2, dep_map3, dep_map4, dep_map5, dep_1_4, dep_1_8, dep_1_16, dep_1_32 = self.depth_backbone(
    #         depth_Input)
    #
    #     img_layer_cat1 = feature_map1 + dep_map1  # img_layer_cat1:(8,3,224,224)
    #     feature_map2, rgb_fea_1_4 = self.rgb_backbone(img_layer_cat1,
    #                                                   layer_flag=2)  # feature_map2:(8,96,56,56) rgb_fea_1_4:(8,3136,96)
    #
    #     img_layer_cat2 = feature_map2 + dep_map2 # img_layer_cat2:(8,6 4,56,56)
    #     feature_map3, rgb_fea_1_8 = self.rgb_backbone(img_layer_cat2,
    #                                                   layer_flag=3)  # feature_map3:(8,64,28,28) rgb_fea_1_8:(8,784,64)
    #
    #     img_layer_cat3 = feature_map3 + dep_map3  # img_layer_cat3:(8,64,28,28)
    #     feature_map4, rgb_fea_1_16 = self.rgb_backbone(img_layer_cat3, layer_flag=4)
    #
    #     img_layer_cat4 = feature_map4 + dep_map4  # img_layer_cat3:(8,64,28,28)
    #     feature_map5, rgb_fea_1_32 = self.rgb_backbone(img_layer_cat4, layer_flag=5)
    #
    #     rgb_fea_1_32, depth_fea_1_32 = self.transformer_32(rgb_fea_1_32,
    #                                                        dep_1_32)  # 更改后 rgb_fea_1_32(b,49,768) depth_fea_1_16(b,49,768)
    #     rgb_fea_1_32 = rgb_fea_1_32.transpose(1, 2).reshape(B, 768, 7, 7)
    #     depth_fea_1_32 = depth_fea_1_32.transpose(1, 2).reshape(B, 768, 7, 7)
    #
    #     outputs = self.decoder.forward(rgb_fea_1_32, depth_fea_1_32, feature_map5, feature_map4, feature_map3,
    #                                    feature_map2, feature_map1)
    #
    #     return outputs
