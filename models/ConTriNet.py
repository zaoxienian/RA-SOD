import torch
import torch.nn as nn
from torch.nn import functional as F
from models.res2net_v1b_base import Res2Net_model
from models.gen_res2net import GenRes2Net_model  


def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RASPM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RASPM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1_0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1_1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2_0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch2_1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3_0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch3_1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1_1(self.branch1_0(x) + x0)
        x2 = self.branch2_1(self.branch2_0(x) + x1)
        x3 = self.branch3_1(self.branch3_0(x) + x2)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x

class MFM_0(nn.Module):
    def __init__(self,in_dim, out_dim):
        super(MFM_0, self).__init__()
        act_fn = nn.ReLU(inplace=True)
        self.layer_10 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, 1, 1), nn.BatchNorm2d(out_dim), act_fn)
        self.layer_21 = nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, 1, 1), nn.BatchNorm2d(out_dim), act_fn)
        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, 3, 1, 1), nn.BatchNorm2d(out_dim), act_fn)
        self.conv2d = nn.Conv2d(2, 1, 7, 1, 3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel_mul_conv1 = nn.Sequential(nn.Conv2d(out_dim, out_dim // 16, 1), nn.ReLU(True), nn.Conv2d(out_dim // 16, out_dim, 1))
        self.channel_mul_conv2 = nn.Sequential(nn.Conv2d(out_dim, out_dim // 16, 1), nn.ReLU(True), nn.Conv2d(out_dim // 16, out_dim, 1))
        
    def forward(self, rgb, thermal):
        x_rgb = self.layer_10(rgb)
        x_the = self.layer_20(thermal)
        rgb_w = self.sigmoid(x_rgb)
        the_w = self.sigmoid(x_the)
        x_rgb_r = self.layer_11((rgb.mul(the_w)) + rgb)
        x_the_r = self.layer_21((thermal.mul(rgb_w)) + thermal)
        x_rgb_r = x_rgb_r * torch.sigmoid(self.channel_mul_conv1(x_rgb_r))
        x_the_r = x_the_r * torch.sigmoid(self.channel_mul_conv2(x_the_r))
        ful_out = torch.cat((x_rgb_r, x_the_r), dim=1)
        mask = self.sigmoid(self.conv2d(torch.cat([torch.mean(ful_out,1,keepdim=True), torch.max(ful_out,1,keepdim=True)[0]], dim=1)))
        return self.layer_ful1(ful_out) * mask

class MFM(nn.Module):
    def __init__(self,in_dim, out_dim):
        super(MFM, self).__init__()
        act_fn = nn.ReLU(inplace=True)
        self.reduc_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1), act_fn)
        self.reduc_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, 1), act_fn)
        self.layer_10 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, 1, 1), nn.BatchNorm2d(out_dim), act_fn)
        self.layer_21 = nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, 1, 1), nn.BatchNorm2d(out_dim), act_fn)
        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, 3, 1, 1), nn.BatchNorm2d(out_dim), act_fn)
        self.layer_ful2 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, 3, 1, 1), nn.BatchNorm2d(out_dim), act_fn)
        self.conv2d = nn.Conv2d(2, 1, 7, 1, 3, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel_mul_conv1 = nn.Sequential(nn.Conv2d(out_dim, out_dim // 16, 1), nn.ReLU(True), nn.Conv2d(out_dim // 16, out_dim, 1))
        self.channel_mul_conv2 = nn.Sequential(nn.Conv2d(out_dim, out_dim // 16, 1), nn.ReLU(True), nn.Conv2d(out_dim // 16, out_dim, 1))

    def forward(self, rgb, thermal, xx):
        x_rgb = self.reduc_1(rgb)
        x_the = self.reduc_2(thermal)
        x_rgb1 = self.layer_10(x_rgb)
        x_the1 = self.layer_20(x_the)
        rgb_w = self.sigmoid(x_rgb1)
        the_w = self.sigmoid(x_the1)
        x_rgb_w = x_rgb.mul(the_w)
        x_the_w = x_the.mul(rgb_w)
        x_rgb_r = self.layer_11(x_rgb_w + x_rgb)
        x_the_r = self.layer_21(x_the_w + x_the)
        x_rgb_r = x_rgb_r * torch.sigmoid(self.channel_mul_conv1(x_rgb_r))
        x_the_r = x_the_r * torch.sigmoid(self.channel_mul_conv2(x_the_r))
        ful_out = torch.cat((x_rgb_r, x_the_r), dim=1)
        avgout = torch.mean(ful_out, dim=1, keepdim=True)
        maxout, _ = torch.max(ful_out, dim=1, keepdim=True)
        mask = self.sigmoid(self.conv2d(torch.cat([avgout, maxout], dim=1)))
        out1 = self.layer_ful1(ful_out) * mask
        out2 = self.layer_ful2(torch.cat([out1,xx],dim=1))
        return out2

class UBiBridge(nn.Module):
    def __init__(self, channels):
        super(UBiBridge, self).__init__()
        
  
        self.unc_head_r = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.unc_head_t = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        
 
        self.trans_r2t = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.trans_t2r = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_rgb, x_the):
 
        unc_r = F.softplus(self.unc_head_r(x_rgb)) 
        unc_t = F.softplus(self.unc_head_t(x_the))
        
 
        gate_import_t = torch.sigmoid(unc_r) 
        gate_import_r = torch.sigmoid(unc_t) 
        
     
        correction_from_t = self.trans_t2r(x_the) * gate_import_t
        out_rgb = x_rgb + correction_from_t
        
        correction_from_r = self.trans_r2t(x_rgb) * gate_import_r
        out_the = x_the + correction_from_r
        
        return out_rgb, out_the, unc_r, unc_t


class SC_URFM(nn.Module):

    def __init__(self, in_dim):
        super(SC_URFM, self).__init__()
        
        
        self.conv_rgb = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )
        self.conv_the = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )
        
        self.reliability_net = nn.Sequential(
            nn.Conv2d(in_dim * 3, in_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, 2, kernel_size=1) 
        )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x_ful, x_rgb, x_thermal):

        feat_r = self.conv_rgb(x_rgb)
        feat_t = self.conv_the(x_thermal)
        

        combined = torch.cat([x_ful, feat_r, feat_t], dim=1)
        spatial_weights = self.reliability_net(combined)
        spatial_weights = F.softmax(spatial_weights, dim=1) 
        
        w_rgb = spatial_weights[:, 0:1, :, :]
        w_the = spatial_weights[:, 1:2, :, :]
        

        fused_features = w_rgb * feat_r + w_the * feat_t
        

        out = self.fusion_conv(fused_features + x_ful)
        
        return out


class MDAM(nn.Module):

    def __init__(self, in_dim):
        super(MDAM, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.t = 30
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.concat_conv1 = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim)
        )
        self.concat_conv2 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim) 
        )
        self.linear_layers = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, stride=1,bias=False),  
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim , kernel_size=1),
        )
        self.dynamic_aggregation = nn.Sequential(
            nn.Linear(in_dim, in_dim // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim // 4, 2, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x_ful, x_rgb, x_thermal):
        x_ful_rgb = x_ful.mul(x_rgb)
        x_ful_thermal = x_ful.mul(x_thermal)
        x_concat = self.concat_conv1(torch.cat([x_ful_rgb, x_ful_thermal], dim=1))

        weight = self.avg_pool(x_concat).view(x_concat.size(0), x_concat.size(1))
        weight = self.dynamic_aggregation(weight)
        weight = F.softmax(weight / self.t, dim=1)

        x_rgb_att = torch.sigmoid(self.linear_layers(x_rgb))
        x_rgb_att = torch.mul(x_concat, x_rgb_att)
        x_thermal_att = torch.mul(x_concat, x_thermal)

        x_att = x_rgb_att * weight[:, 0].view(x_concat.size(0), 1, 1, 1) + x_thermal_att * weight[:, 1].view(x_concat.size(0), 1, 1, 1)
        out = self.concat_conv2(x_att)
        out = self.relu(out) + x_ful
        return out


class ConTriNet_R50(nn.Module):
    def __init__(self, channel=64, use_gen_backbone=True):
        super(ConTriNet_R50, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.use_gen_backbone = use_gen_backbone


        if use_gen_backbone:
           
            self.backbone = GenRes2Net_model(ind=50)
        else:
            self.layer_rgb = self.layer_the = Res2Net_model(ind=50)
            
        self.layer_the0 = nn.Conv2d(1, 3, kernel_size=1)

  
        self.fu_0 = MFM_0(64, 64)
        self.fu_1 = MFM(256, 64)
        self.fu_2 = MFM(512, 64)
        self.fu_3 = MFM(1024, 64)
        self.fu_4 = MFM(2048, 64)

        self.pool_fu_1 = maxpool()
        self.pool_fu_2 = maxpool()
        self.pool_fu_3 = maxpool()


        self.rgb_gcm_4 = RASPM(2048, channel)
        self.rgb_gcm_3 = RASPM(1024 + channel, channel)
        self.rgb_gcm_2 = RASPM(512 + channel, channel)
        self.rgb_gcm_1 = RASPM(256 + channel, channel)
        self.rgb_gcm_0 = RASPM(64 + channel, channel)
        self.rgb_conv_out = nn.Conv2d(channel, 1, 1)

       
        self.the_gcm_4 = RASPM(2048, channel)
        self.the_gcm_3 = RASPM(1024 + channel, channel)
        self.the_gcm_2 = RASPM(512 + channel, channel)
        self.the_gcm_1 = RASPM(256 + channel, channel)
        self.the_gcm_0 = RASPM(64 + channel, channel)
        self.the_conv_out = nn.Conv2d(channel, 1, 1)

        
        self.ful_gcm_4 = RASPM(channel, channel)
        self.ful_gcm_3 = RASPM(channel + channel, channel)
        self.ful_gcm_2 = RASPM(channel + channel, channel)
        self.ful_gcm_1 = RASPM(channel + channel, channel)
        self.ful_gcm_0 = RASPM(channel + channel, channel)
        self.ful_conv_out = nn.Conv2d(channel, 1, 1)

       
        self.bridge_4 = UBiBridge(channel)
        self.bridge_3 = UBiBridge(channel)
        self.bridge_2 = UBiBridge(channel)
        self.bridge_1 = UBiBridge(channel)

      
        self.ful_layer3 = SC_URFM(channel)
        self.ful_layer2 = SC_URFM(channel)
        self.ful_layer1 = SC_URFM(channel)
        self.ful_layer0 = SC_URFM(channel)
        

    def forward(self, imgs, thermals):
        

        if self.use_gen_backbone:
            img_0, img_1, img_2, img_3, img_4, rgb_gates = self.backbone(imgs, modality='rgb')
            the_0, the_1, the_2, the_3, the_4, the_gates = self.backbone(self.layer_the0(thermals), modality='thermal')
            self.rgb_gates, self.the_gates = rgb_gates, the_gates
        else:
            img_0, img_1, img_2, img_3, img_4 = self.layer_rgb(imgs)
            the_0, the_1, the_2, the_3, the_4 = self.layer_the(self.layer_the0(thermals))

        ful_0 = self.fu_0(img_0, the_0) 
        ful_1 = self.fu_1(img_1, the_1, ful_0)
        ful_2 = self.fu_2(img_2, the_2, self.pool_fu_1(ful_1))
        ful_3 = self.fu_3(img_3, the_3, self.pool_fu_2(ful_2))
        ful_4 = self.fu_4(img_4, the_4, self.pool_fu_3(ful_3)) 

        self.unc_maps = [] 

        x_rgb_4 = self.rgb_gcm_4(img_4)
        x_the_4 = self.the_gcm_4(the_4)
        
        x_rgb_4, x_the_4, unc_r4, unc_t4 = self.bridge_4(x_rgb_4, x_the_4)
        self.unc_maps.append((unc_r4, unc_t4))

        x_ful_4 = self.ful_gcm_4(ful_4)
        
        x_rgb_3 = self.rgb_gcm_3(torch.cat([img_3, F.interpolate(x_rgb_4, scale_factor=2, mode='bilinear', align_corners=True)], dim=1))
        x_the_3 = self.the_gcm_3(torch.cat([the_3, F.interpolate(x_the_4, scale_factor=2, mode='bilinear', align_corners=True)], dim=1))
        
 
        x_rgb_3, x_the_3, unc_r3, unc_t3 = self.bridge_3(x_rgb_3, x_the_3)
        self.unc_maps.append((unc_r3, unc_t3))

        sc_feat_3 = self.ful_layer3(
            F.interpolate(x_ful_4, scale_factor=2, mode='bilinear', align_corners=True), 
            x_rgb_3, 
            x_the_3
        )
        x_ful_3 = self.ful_gcm_3(torch.cat([ful_3, sc_feat_3], dim=1))

        x_rgb_2 = self.rgb_gcm_2(torch.cat([img_2, F.interpolate(x_rgb_3, scale_factor=2, mode='bilinear', align_corners=True)], dim=1))
        x_the_2 = self.the_gcm_2(torch.cat([the_2, F.interpolate(x_the_3, scale_factor=2, mode='bilinear', align_corners=True)], dim=1))
        
        x_rgb_2, x_the_2, unc_r2, unc_t2 = self.bridge_2(x_rgb_2, x_the_2)
        self.unc_maps.append((unc_r2, unc_t2))

        sc_feat_2 = self.ful_layer2(
            F.interpolate(x_ful_3, scale_factor=2, mode='bilinear', align_corners=True),
            x_rgb_2,
            x_the_2
        )
        x_ful_2 = self.ful_gcm_2(torch.cat([ful_2, sc_feat_2], dim=1))

        x_rgb_1 = self.rgb_gcm_1(torch.cat([img_1, F.interpolate(x_rgb_2, scale_factor=2, mode='bilinear', align_corners=True)], dim=1))
        x_the_1 = self.the_gcm_1(torch.cat([the_1, F.interpolate(x_the_2, scale_factor=2, mode='bilinear', align_corners=True)], dim=1))
        
        x_rgb_1, x_the_1, unc_r1, unc_t1 = self.bridge_1(x_rgb_1, x_the_1)
        self.unc_maps.append((unc_r1, unc_t1))

        sc_feat_1 = self.ful_layer1(
            F.interpolate(x_ful_2, scale_factor=2, mode='bilinear', align_corners=True),
            x_rgb_1,
            x_the_1
        )
        x_ful_1 = self.ful_gcm_1(torch.cat([ful_1, sc_feat_1], dim=1))

        h0, w0 = img_0.size()[2:] 
        
        x_rgb_1_up = F.interpolate(x_rgb_1, size=(h0, w0), mode='bilinear', align_corners=True)
        x_rgb_0 = self.rgb_gcm_0(torch.cat([img_0, x_rgb_1_up], dim=1))
        
        x_the_1_up = F.interpolate(x_the_1, size=(h0, w0), mode='bilinear', align_corners=True)
        x_the_0 = self.the_gcm_0(torch.cat([the_0, x_the_1_up], dim=1))
        
        x_ful_1_up = F.interpolate(x_ful_1, size=(h0, w0), mode='bilinear', align_corners=True)
        
        sc_feat_0 = self.ful_layer0(
            x_ful_1_up, 
            x_rgb_0, 
            x_the_0
        )
        x_ful_0 = self.ful_gcm_0(torch.cat([ful_0, sc_feat_0], dim=1))

        rgb_out = F.interpolate(self.rgb_conv_out(x_rgb_0), scale_factor=4, mode='bilinear', align_corners=True)
        the_out = F.interpolate(self.the_conv_out(x_the_0), scale_factor=4, mode='bilinear', align_corners=True)
        ful_out = F.interpolate(self.ful_conv_out(x_ful_0), scale_factor=4, mode='bilinear', align_corners=True)

        out = rgb_out + the_out + ful_out

        return out, rgb_out, the_out, ful_out

    def get_gate_weights(self):
        if self.use_gen_backbone:
            return self.rgb_gates, self.the_gates
        else:
            return None, None

    def get_uncertainty_maps(self):
        return self.unc_maps


if __name__ == '__main__':
    print("Initializing Gen-DURNet (ConTriNet_R50)...")
    model = ConTriNet_R50(channel=64, use_gen_backbone=True)
    
    rgb = torch.rand(2, 3, 352, 352)
    thermal = torch.rand(2, 1, 352, 352)
    
    if torch.cuda.is_available():
        model = model.cuda()
        rgb = rgb.cuda()
        thermal = thermal.cuda()

    model.eval()
    with torch.no_grad():
        out, rgb_out, the_out, ful_out = model(rgb, thermal)
    
    print(f"Output shape: {out.shape}")
    print("Test passed successfully.")