import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools


from .darknet import BaseConv, CSPDarknet, CSPLayer, DWConv, get_activation
from .BiConvLSTM import BiConvLSTM
from nets.ops.dcn.deform_conv import ModulatedDeformConv


# from darknet import BaseConv, CSPDarknet, CSPLayer, DWConv, get_activation
# from BiConvLSTM import BiConvLSTM



class Feature_Extractor(nn.Module):
    def __init__(self, depth = 1.0, width = 1.0, in_features = ("dark3", "dark4", "dark5"), in_channels = [256, 512, 1024], depthwise = False, act = "silu"):
        super().__init__()
        Conv                = DWConv if depthwise else BaseConv
        self.backbone       = CSPDarknet(depth, width, depthwise = depthwise, act = act)
        self.in_features    = in_features

        self.upsample       = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.lateral_conv0  = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)
    
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )  

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.reduce_conv1   = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise = depthwise,
            act = act,
        )


    def forward(self, input):
        out_features            = self.backbone.forward(input)
        [feat1, feat2, feat3]   = [out_features[f] for f in self.in_features]

        #-------------------------------------------#
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.lateral_conv0(feat3)
        #-------------------------------------------#
        #  20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #  40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.C3_p4(P5_upsample)

        #-------------------------------------------#
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.reduce_conv1(P5_upsample) 
        #-------------------------------------------#
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4) 
        #-------------------------------------------#
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample = torch.cat([P4_upsample, feat1], 1) 
        #-------------------------------------------#
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out      = self.C3_p3(P4_upsample)  
        return P3_out
    


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [16, 32, 64], act = "silu"):
        super().__init__()
        Conv            =  BaseConv
        
        self.stems      = nn.ModuleList()

        self.cls_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
    
        self.reg_convs  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()

        self.obj_preds  = nn.ModuleList()
        headnf = int(256 * width)

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = headnf, ksize = 1, stride = 1, act = act))
            
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = headnf, out_channels = headnf, ksize = 3, stride = 1, act = act), 
                Conv(in_channels = headnf, out_channels = headnf, ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = headnf, out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )

            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = headnf, out_channels = headnf, ksize = 3, stride = 1, act = act), 
                Conv(in_channels = headnf, out_channels = headnf, ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = headnf, out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )

            self.obj_preds.append(
                nn.Conv2d(in_channels = headnf, out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):   # B, C, H, W
        # outputs = []
        for k, x in enumerate(inputs):
            x = self.stems[k](x)

            cls_feat    = self.cls_convs[k](x)
            cls_output  = self.cls_preds[k](cls_feat)  # cls_output: B, num_classes, H, W

            reg_feat    = self.reg_convs[k](x)
            reg_output  = self.reg_preds[k](reg_feat)  # reg_output: B, 4, H, W

            obj_output  = self.obj_preds[k](reg_feat)  # cls_output: B, 1, H, W

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
        #     outputs.append(output)
        # return outputs
        return output






class CA_block(nn.Module):   
    def __init__(self, in_channel=32, reduce_ratio=4):
        super(CA_block, self).__init__()
        self.ca_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=in_channel, out_channels=in_channel // reduce_ratio, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=in_channel // reduce_ratio, out_channels=in_channel, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid()
        )
    
    def forward(self, x):
        x1 = self.ca_layer(x)
        x = x * x1
        return x
    

class Ada_RDBlock(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, reduce_ratio=4, a=1, b=0.2):
        super(Ada_RDBlock, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate))
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.ca_block = CA_block(in_channel=in_channels_, reduce_ratio=reduce_ratio)
        # self.conv3x3 = nn.Conv2d(in_channels=in_channels_, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.conv3x3 = nn.Conv2d(in_channels=in_channels_, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.fuse_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.fuse_weight.data.fill_(a)
        self.fuse_weight_1.data.fill_(b)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.ca_block(out)
        out = self.conv3x3(out)
        return x * self.fuse_weight + out * self.fuse_weight_1


class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate):
        super(dense_layer, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=growthRate, kernel_size=3, stride=1, padding=1)   
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out








class Channel_Attention(nn.Module):
    def __init__(self, in_nc, ratio=16):
        super(Channel_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_nc, in_nc // ratio, 1, bias=False), 
            nn.ReLU(),
            nn.Conv2d(in_nc // ratio, in_nc, 1, bias=False)
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout) * x
    

class Align_Net(nn.Module):   
    def __init__(self, in_nc, out_nc, base_ks=3, deform_nc=256, deform_ks=3, deform_group=8):
        super(Align_Net, self).__init__()
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.deform_group = deform_group
        self.offset_mask = nn.Conv2d(in_nc, deform_group*3*(deform_ks**2), base_ks, padding=base_ks//2)
        self.deform_conv = ModulatedDeformConv(deform_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=deform_group)
    
    def forward(self, x, y):
        off_msk = self.offset_mask(x)


        off_msk = F.interpolate(off_msk, (y.size(2), y.size(3)), mode='bilinear', align_corners=False)


        off = off_msk[:, :self.deform_group*2*(self.deform_ks**2), ...]
        msk = torch.sigmoid(off_msk[:, self.deform_group*2*(self.deform_ks**2):, ...])

        fused_feat = F.relu(self.deform_conv(y, off, msk), inplace=True)
        return fused_feat
    

class Network(nn.Module):
    def __init__(self, num_classes, fp16=False, num_frame=10, training=True, mobile_sam=None):
        super(Network, self).__init__()
        self.num_frame = num_frame
        act = "silu"
        self.head_nf = 128
        self.mobile_sam = mobile_sam

        # fea_ext_nf = 64
        fea_ext_out_nc = 64
        # self.fea_ext = Feature_Extractor(in_nc=3, nf=fea_ext_nf, out_nc=fea_ext_out_nc, act=act) 
        self.backbone = Feature_Extractor(0.33, 0.50) 
        self.fe_conv = BaseConv(128, fea_ext_out_nc, 3, 1, act=act)
        #----------------------------------------------------------#
        #   head
        #----------------------------------------------------------#
        self.head = YOLOXHead(num_classes=num_classes, width=1.0, in_channels=[self.head_nf], act=act)

        input_size = [64, 64]
        self.lstm = BiConvLSTM(input_size=(input_size[0], input_size[1]), input_dim=fea_ext_out_nc, 
                             hidden_dim=fea_ext_out_nc, kernel_size=(3, 3), num_layers=1)

        self.concat_conv = nn.Sequential(
            BaseConv(fea_ext_out_nc*5, fea_ext_out_nc, 3, 1, act=act),
            Channel_Attention(fea_ext_out_nc, ratio=8),
            # BaseConv(fea_ext_out_nc, self.head_nf, 3, 1, act=act),
            BaseConv(fea_ext_out_nc, fea_ext_out_nc, 3, 1, act=act),
        )
        self.conv_fuse = nn.Sequential(
            BaseConv(fea_ext_out_nc*2, self.head_nf, 3, 1, act=act),
        )

        ###################################
        self.mobile_sam = mobile_sam
        self.head = YOLOXHead(num_classes=num_classes, width=1.0, in_channels=[self.head_nf])
        self.fuse_source_target = nn.Sequential(
            BaseConv(256 * 2, 128, 3, 1, act=act),
            BaseConv(128, 64, 3, 1, act=act),
            self.make_layer(functools.partial(Ada_RDBlock, 64, 64, 3, 4, 1, 0.2), 3),
            )
        self.align_source_target = Align_Net(in_nc=64, out_nc=64, deform_nc=256, deform_group=8)
        self.align_conv_source_target = nn.Sequential(
            BaseConv(64, 128, 3, 1, act=act),
            BaseConv(128, 256, 3, 1, act=act),
            )
        self.mask_conv = nn.Sequential(
            BaseConv(1, 64, 3, 1, act=act),
            BaseConv(64, self.head_nf, 3, 2, act=act),
            BaseConv(self.head_nf, self.head_nf, 3, 1, act=act),
        )

        ##########################
        self.final_conv = nn.Sequential(
            BaseConv(self.head_nf*2, self.head_nf, 3, 1, act=act),
            Channel_Attention(self.head_nf, ratio=8),
            BaseConv(self.head_nf, self.head_nf, 3, 1, act=act)
        )

    def forward(self, source_image, raw_source_box, target_images): #4, 3, 5, 512, 512
        b, c, t, h, w = target_images.shape
        
        source_image_embedding = self.mobile_sam.image_encoder(source_image)
        # print(source_image_embedding.shape)  # torch.Size([b, 256, h//16, w//16]) 
        # source_image_pe = self.mobile_sam.prompt_encoder.get_dense_pe()
        # print(source_image_pe.shape)  # torch.Size([1, 256, h//16, w//16])


        target_image_embedding = self.mobile_sam.image_encoder(target_images[:, :, -1, :, :])
        target_image_pe = self.mobile_sam.prompt_encoder.get_dense_pe()  
        # print(target_image_embedding.shape, target_image_pe.shape)  # torch.Size([b, 256, h//16, w//16]) torch.Size([1, 256, h//16, w//16])
        
        cat_image_embedding = torch.cat((source_image_embedding, target_image_embedding), dim=1)
        cat_image_embedding = self.fuse_source_target(cat_image_embedding)
        # print(self.align(x, y).shape)


        cat_source_sparse_embeddings = []
        for bidx in range(b):
            source_sparse_embeddings, dense_embeddings = self.mobile_sam.prompt_encoder(
                points=None,
                boxes=raw_source_box[bidx],
                masks=None,
            )

            # print(source_sparse_embeddings.shape, dense_embeddings.shape)  # torch.Size([框数, 2, 256]) torch.Size([框数, 256, h//16, w//16])
            source_sparse_embeddings = source_sparse_embeddings.view(-1, 256)  # torch.Size([框数*2, 256])
            cat_source_sparse_embeddings.append(source_sparse_embeddings)
        cat_source_sparse_embeddings = torch.stack(cat_source_sparse_embeddings, dim=0)  # torch.Size([b, 框数*2, 256])
        cat_source_sparse_embeddings = cat_source_sparse_embeddings.permute(0, 2, 1)  # torch.Size([b, 256, 框数*2])
        cat_source_sparse_embeddings = cat_source_sparse_embeddings.unsqueeze(-1)  # torch.Size([b, 256, 框数*2, 1])
        cat_source_sparse_embeddings = cat_source_sparse_embeddings.repeat(1, 1, 1, cat_source_sparse_embeddings.shape[2])  # torch.Size([b, 256, 框数*2, 框数*2])
        
        cat_target_sparse_embeddings = self.align_conv_source_target(self.align_source_target(cat_image_embedding, cat_source_sparse_embeddings))  # torch.Size([b, 256, 框数*2, 框数*2])
        cat_target_sparse_embeddings = torch.mean(cat_target_sparse_embeddings, dim=-1).permute(0, 2, 1)  # torch.Size([b, 框数*2, 256])
        cat_target_sparse_embeddings = cat_target_sparse_embeddings.view(b, -1, 2, 256)  # torch.Size([b, 框数, 2, 256])


        cat_mask_predictions = []
        for bidx in range(b):
            sparse_embeddings = cat_target_sparse_embeddings[bidx, ...]  

            mask_predictions, iou_preds = self.mobile_sam.mask_decoder(
                image_embeddings=target_image_embedding[bidx:bidx+1, ...],  # (1, 256, h//16, w//16)
                image_pe=target_image_pe,  # torch.Size([1, 256, 64, 64])
                sparse_prompt_embeddings=sparse_embeddings,  # (框数, 2, 256)
                dense_prompt_embeddings=dense_embeddings,  # (框数, 256, h//16, w//16)
                multimask_output=False,
            )  
            # print(mask_predictions.shape, iou_preds.shape)  # torch.Size([框数, 1, h//4, w//4]), torch.Size([框数, 1])
            # exit(1)

            cat_mask_predictions.append(mask_predictions)
        
        cat_mask_predictions = torch.stack(cat_mask_predictions, dim=0)  # torch.Size([b, 框数, 1, h//4, w//4])
        _, box_num, _, _, _ = cat_mask_predictions.shape
        


        outputs = []
        ####################################

        feat_list = []
        for i in range(t):
            feat_list.append(self.fe_conv(self.backbone(target_images[:, :, i, :, :])))
        
        lstm_input      = torch.stack(feat_list, dim=1)
        lstm_output = self.lstm(lstm_input)  
        # print(lstm_output.shape)  # torch.Size([4, 5, 64, 64, 64])


        concat_input = torch.cat([lstm_output[:, i, :, :, :] for i in range(t)], dim=1)
        out_feat = self.concat_conv(concat_input)
        out_feat = self.conv_fuse( torch.cat( (out_feat, feat_list[-1]), dim=1) )




        ####################################融合
        for i in range(box_num):
            feat = self.mask_conv(cat_mask_predictions[:, i, :, :, :])  # torch.Size([4, 128, 64, 64]) torch.Size([4, 1, 128, 128])
            # print(out_feat.shape, feat.shape)  # torch.Size([4, 128, 64, 64]) torch.Size([4, 128, 64, 64])
            
            feat = torch.cat((out_feat, feat), dim=1)
            feat = self.final_conv(feat)


            output = self.head([feat])
            outputs.append(output)
        return outputs
            
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    
if __name__ == "__main__":
    
    from yolo_training import YOLOLoss
    net = Network(num_classes=1, num_frame=5).cuda(0)


    bs = 2
    target_images = torch.randn(bs, 3, 5, 512, 512).cuda(0) 
    # out = net(a) 


    source_image = torch.rand(bs, 3, 512, 512).cuda(0)
    raw_source_box = torch.rand(bs, 1, 4).cuda(0)

    from thop import profile
    flops, params = profile(net, inputs=(source_image, raw_source_box, target_images, ))
    print(flops/(10**9), params/(10**6))  





    # for item in out:
    #     print(item.size())
        
    # yolo_loss    = YOLOLoss(num_classes=1, fp16=False, strides=[16])

    # target = torch.randn([bs, 1, 5]).cuda()
    # target = nn.Softmax()(target)
    # target = [item for item in target]

    # loss = yolo_loss(out, target)
    # print(loss)
