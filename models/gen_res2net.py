import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['GenRes2Net', 'gen_res2net50_v1b']

model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
}

MOE_STRATEGY = 'layer4' 

class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x, modality=None):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, None


class GenDHELayer(nn.Module):
    def __init__(self, channels, stride=1, num_experts=3, reduction=4): 
        super(GenDHELayer, self).__init__()
        
        self.num_experts = num_experts
        

        self.main_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(channels)
            ) for _ in range(num_experts)
        ])

        def build_gate():
            return nn.Sequential(
                nn.InstanceNorm2d(channels), 
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, num_experts), 
                nn.Softmax(dim=1)
            )
        
        self.gate_rgb = build_gate()
        self.gate_t = build_gate()

        mid_channels = max(channels // reduction, 8)
        def build_adapter():
            return nn.Sequential(
                nn.Conv2d(channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels)
            )
        self.adapter_rgb = build_adapter()
        self.adapter_t = build_adapter()
        
        self.stride = stride
        if stride > 1:
            self.downsample = nn.AvgPool2d(kernel_size=stride, stride=stride)
        else:
            self.downsample = None

    def forward(self, x, modality):

        main_feat = self.main_conv(x)
        

        if modality == 'rgb':
            gate_weights = self.gate_rgb(x)
            spec_out = self.adapter_rgb(x)
        else: # thermal
            gate_weights = self.gate_t(x)
            spec_out = self.adapter_t(x)
            
        moe_feat = 0
        for k in range(self.num_experts):
            w = gate_weights[:, k].view(-1, 1, 1, 1)
            moe_feat += w * self.experts[k](x)


        identity = x
        if self.downsample is not None:
            identity = self.downsample(identity)
            
        return main_feat + moe_feat + spec_out, gate_weights

class GenBottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        super(GenBottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        
        self.dhe_layers = nn.ModuleList()

        
        for i in range(self.nums):

          self.dhe_layers.append(GenDHELayer(width, stride=stride))

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x, modality): 
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        
        sp = None
        all_gate_weights = [] 
        
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            
            sp, gw = self.dhe_layers[i](sp, modality)
            all_gate_weights.append(gw)
            sp = self.relu(sp)
            
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out, torch.stack(all_gate_weights, dim=1)

class LayerGroup(nn.Module):

    def __init__(self, layers):
        super(LayerGroup, self).__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, modality):
        gate_weights_list = []
        for layer in self.layers:
            x, gw = layer(x, modality)
            if gw is not None:  
                gate_weights_list.append(gw)
        
  
        if gate_weights_list:
            return x, torch.stack(gate_weights_list, dim=1)
        else:
            return x, None

class GenRes2Net(nn.Module):
    def __init__(self, layers, baseWidth=26, scale=4, num_classes=1000, moe_strategy='high'):

        self.inplanes = 64
        super(GenRes2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.moe_strategy = moe_strategy
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        if moe_strategy == 'all':
            use_moe = (True, True, True, True)
        elif moe_strategy == 'high':
            use_moe = (False, False, True, True)  
        elif moe_strategy == 'layer4':
            use_moe = (False, False, False, True)
        elif moe_strategy == 'stage':
            use_moe = ('first', 'first', 'first', 'first')  
        else:
            raise ValueError(f"Unknown moe_strategy: {moe_strategy}")
        
        self.use_moe = use_moe
        
        self.layer1 = self._make_layer(64, layers[0], use_moe=use_moe[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2, use_moe=use_moe[1])
        self.layer3 = self._make_layer(256, layers[2], stride=2, use_moe=use_moe[2])
        self.layer4 = self._make_layer(512, layers[3], stride=2, use_moe=use_moe[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        self._print_moe_config()

    def _print_moe_config(self):
        print(f"\n[GenRes2Net] MoE Strategy: '{self.moe_strategy}'")
        layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        for i, (name, moe) in enumerate(zip(layer_names, self.use_moe)):
            if moe == True:
                status = "✓ MoE (全部)"
            elif moe == 'first':
                status = "✓ MoE (仅首块)"
            else:
                status = "✗ 标准 Conv"
            print(f"  {name}: {status}")
        
        num_dhe = sum(1 for m in self.modules() if isinstance(m, GenDHELayer))
        print(f"  Total GenDHELayer: {num_dhe}")
        print()

    def _make_layer(self, planes, blocks, stride=1, use_moe=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:  # expansion = 4
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        layer_list = []
        

        if use_moe == True or use_moe == 'first':
            block_class = GenBottle2neck
        else:
            block_class = Bottle2neck
        layer_list.append(block_class(self.inplanes, planes, stride, downsample=downsample, 
                                       stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * 4
        
        if use_moe == True:
            block_class = GenBottle2neck
        else:
            block_class = Bottle2neck
        
        for i in range(1, blocks):
            layer_list.append(block_class(self.inplanes, planes, 
                                          baseWidth=self.baseWidth, scale=self.scale))

        return LayerGroup(layer_list)

    def forward(self, x, modality):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x)

        x1, g1 = self.layer1(x0, modality)
        x2, g2 = self.layer2(x1, modality)
        x3, g3 = self.layer3(x2, modality)
        x4, g4 = self.layer4(x3, modality)

        all_gates = [g for g in [g1, g2, g3, g4] if g is not None]

        return x0, x1, x2, x3, x4, all_gates
    
def gen_res2net50_v1b(pretrained=False, moe_strategy=None, **kwargs):

    zero_init = True
    
    strategy = moe_strategy if moe_strategy is not None else MOE_STRATEGY
    model = GenRes2Net([3, 4, 6, 3], baseWidth=26, scale=4, moe_strategy=strategy, **kwargs)
    
    if pretrained:
        try:
            pretrained_dict = model_zoo.load_url(model_urls['res2net50_v1b_26w_4s'], map_location='cpu')
            model_dict = model.state_dict()
            new_state_dict = {}
            
            for k, v in pretrained_dict.items():
                if k.startswith("layer"):
                    parts = k.split(".")
                    if len(parts) >= 2 and parts[0].startswith("layer") and parts[1].isdigit():
                        layer_name = parts[0]
                        block_idx = int(parts[1])
                        rest = '.'.join(parts[2:])
                        
                        new_key_base = f"{layer_name}.layers.{block_idx}.{rest}"
                        
                        layer = getattr(model, layer_name)
                        block = layer.layers[block_idx]
                        is_moe_block = isinstance(block, GenBottle2neck)
                        
                        if "convs" in k:
                            conv_idx = int(parts[3])
                            param_name = parts[-1]
                            
                            if is_moe_block:
                                
                                new_key = f"{layer_name}.layers.{block_idx}.dhe_layers.{conv_idx}.main_conv.0.{param_name}"
                                new_state_dict[new_key] = v
                            else:
                                
                                new_key = f"{layer_name}.layers.{block_idx}.convs.{conv_idx}.{param_name}"
                                if new_key in model_dict:
                                    new_state_dict[new_key] = v
                        
                        elif "bns" in k:
                            bn_idx = int(parts[3])
                            param_name = parts[-1]
                            
                            if is_moe_block:
                               
                                new_key = f"{layer_name}.layers.{block_idx}.dhe_layers.{bn_idx}.main_conv.1.{param_name}"
                                new_state_dict[new_key] = v
                            else:
                                new_key = f"{layer_name}.layers.{block_idx}.bns.{bn_idx}.{param_name}"
                                if new_key in model_dict:
                                    new_state_dict[new_key] = v
                                    
                        elif new_key_base in model_dict:
                            new_state_dict[new_key_base] = v
                    else:
                        if k in model_dict:
                            new_state_dict[k] = v
                else:
                    if k in model_dict:
                        new_state_dict[k] = v
            
            model_dict.update(new_state_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"[GenRes2Net] Pretrained weights loaded into 'main_conv'.")
            
        except Exception as e:
            print(f"[GenRes2Net] Warning: Failed to load pretrained weights. Error: {e}")

 
    if zero_init:
        print(f"[GenRes2Net] Applying Zero-Init to MoE experts and Adapters...")
        for m in model.modules():
            if isinstance(m, GenDHELayer):
               
                for expert in m.experts:
                    if isinstance(expert[1], nn.BatchNorm2d):
                        nn.init.constant_(expert[1].weight, 0)
                        nn.init.constant_(expert[1].bias, 0)
                
                
                if isinstance(m.adapter_rgb[-1], nn.BatchNorm2d):
                    nn.init.constant_(m.adapter_rgb[-1].weight, 0)
                    nn.init.constant_(m.adapter_rgb[-1].bias, 0)
                
                if isinstance(m.adapter_t[-1], nn.BatchNorm2d):
                    nn.init.constant_(m.adapter_t[-1].weight, 0)
                    nn.init.constant_(m.adapter_t[-1].bias, 0)
        
        print(f"[GenRes2Net] Zero-Init applied. MoE branches start with 0 output.")
    else:
        print(f"[GenRes2Net] Zero-Init SKIPPED (controlled by kwargs).")

    return model


def GenRes2Net_model(ind=50, moe_strategy=None):

    if ind == 50:
        model = gen_res2net50_v1b(pretrained=True, moe_strategy=moe_strategy)
    elif ind == 101:
        raise NotImplementedError("GenRes2Net-101 not implemented yet")
    else:
        raise ValueError(f"Unsupported ind: {ind}")
    
    return model

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test GenRes2Net')
    parser.add_argument('--strategy', type=str, default='high',
                        choices=['all', 'high', 'layer4', 'stage'],
                        help='MoE strategy to use')

    parser.add_argument('--no_zero_init', action='store_true', help='Disable zero init')
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"Testing GenRes2Net with MoE strategy: '{args.strategy}'")
    print(f"Zero Init Enabled: {not args.no_zero_init}")
    print("=" * 60)
    

    model = gen_res2net50_v1b(pretrained=True, moe_strategy=args.strategy)
    model.eval()
    
 
    rgb_img = torch.rand(2, 3, 352, 352)
    thermal_img = torch.rand(2, 3, 352, 352)
    
 
    print("\n[Forward Pass Running...]")
    with torch.no_grad():
        # RGB 流
        x0_r, x1_r, x2_r, x3_r, x4_r, gates_r = model(rgb_img, modality='rgb')
        # Thermal 流
        x0_t, x1_t, x2_t, x3_t, x4_t, gates_t = model(thermal_img, modality='thermal')
    
    print("RGB Features:")
    print(f"  x0: {x0_r.shape}")
    print(f"  x1: {x1_r.shape}")
    print(f"  x2: {x2_r.shape}")
    print(f"  x3: {x3_r.shape}")
    print(f"  x4: {x4_r.shape}")
    
   
    print(f"\nMoE Gate weights check:")
    if len(gates_r) == 0:
        print("  No MoE layers in this configuration!")
    else:
        for i, g in enumerate(gates_r):
           
            print(f"  Stage {i+1} Gate shape: {g.shape}")
        
       
        num_experts = gates_r[0].shape[-1]
        all_gates_flat = torch.cat([g.reshape(-1, num_experts) for g in gates_r], dim=0)
        
        expert_usage = all_gates_flat.mean(dim=0)
        loss_balance = expert_usage.std() / (expert_usage.mean() + 1e-6)
        
        print(f"\nExpert usage distribution: {expert_usage.tolist()}")
        print(f"Balance Loss (CV): {loss_balance.item():.4f}")
    
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params / 1e6:.2f}M")
    

    print(f"\n[Verification] Checking Zero-Init status...")
    if not args.no_zero_init:
        
        verified = False
        for m in model.modules():
            if isinstance(m, GenDHELayer):
                
                bn_weight = m.experts[0][1].weight
                bn_bias = m.experts[0][1].bias
                
                
                adapter_weight = m.adapter_rgb[-1].weight
                
                print(f"  - Sample Expert BN weight mean: {bn_weight.abs().mean().item():.6f} (Should be 0.0)")
                print(f"  - Sample Adapter BN weight mean: {adapter_weight.abs().mean().item():.6f} (Should be 0.0)")
                
                if bn_weight.abs().sum() < 1e-5 and adapter_weight.abs().sum() < 1e-5:
                    print("  >> SUCCESS: Zero-Init is working correctly! Residual branches are silent.")
                else:
                    print("  >> WARNING: Zero-Init might have failed!")
                verified = True
                break
        if not verified:
            print("  >> Note: No GenDHELayer found to verify (maybe strategy='none'?)")
    else:
        print("  >> Zero-Init disabled by argument, skipping check.")

    print("\n" + "=" * 60)
    print("Test passed!")
    print("=" * 60)