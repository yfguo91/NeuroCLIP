import torch
import torch.nn as nn

from dassl.engine import TRAINER_REGISTRY, TrainerX

from clip import clip
from .fewshot import load_clip_to_cpu
from trainers.mv_utils_zs import PCViews
from IPython import embed
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

CUSTOM_TEMPLATES = {
    'NMNIST_DATA': 'A black and white photo of the Arabic numeral number: "{}".',
    'CIFAR10DVS_DATA': 'A low quality stick figure of the object {}.',
    'ESIMAGENET_DATA': 'A low quality point cloud image of {}.'
}


class Textual_Encoder(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self):
        temp = CUSTOM_TEMPLATES[self.cfg.DATASET.NAME]
        prompts = [temp.format(c.replace('_', ' ')) for c in self.classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.cuda()
        text_feat = self.clip_model.encode_text(prompts).repeat(1, self.cfg.MODEL.PROJECT.NUM_TIMES)
        return text_feat


@TRAINER_REGISTRY.register()
class NeuroCLIP_ZS(TrainerX):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.cuda()

        # Encoders from CLIP
        self.visual_encoder = clip_model.visual
        self.textual_encoder = Textual_Encoder(cfg, classnames, clip_model)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.channel = cfg.MODEL.BACKBONE.CHANNEL
    
        self.num_times = cfg.MODEL.PROJECT.NUM_TIMES


        # Store features for post-process view-weight search
        self.feat_store = []
        self.label_store = []

    def mv_proj(self, nc):
        b, t, c, h, w = nc.shape
        #img = torch.sum(nc, 1, keepdim=False)
        
        output_img = (torch.ones(b, t, h, w) * 127).cuda()
        #output_img = (torch.ones(b, 3, h, w) * 127).cuda()
        
        #torch.nn.functional.upsample(nc, size=(224, 224), mode='bilinear')
        for j in range(t):
            img = nc[:, j, : , :, :]
            for i in range(2):
                tmp = img[:,i,:,:]
                if i == 0:
                    tmp_i = torch.where(tmp > 0)    #for cifar10-dvs, please change to tmp > 1
                    output_img[:,j,:,:][tmp_i] = 255
                else:
                    tmp_i = torch.where(tmp > 0)    #for cifar10-dvs, please change to tmp > 1
                    output_img[:,j,:,:][tmp_i] = 0
                #output_img += tmp_i.unsqueeze(1).repeat(1, 3, 1, 1)
                
                #output_img += tmp_i.unsqueeze(1).repeat(1, 3, 1, 1)  
        #output_img = torch.nn.functional.upsample(output_img, size=(224, 224), mode='nearest')
        #output_img = torch.nn.functional.upsample(output_img, size=(48, 48), mode='bilinear')
        output_img = torch.nn.functional.upsample(output_img, size=(224, 224), mode='bilinear')
        output_img = output_img.unsqueeze(2).repeat(1, 1, 3, 1, 1)     
        return output_img


    
    def model_inference(self, nc, label=None):

        # generate multi  maps
        
        images = self.mv_proj(nc).type(self.dtype)
        B,T,C,W,H = images.shape
        images = images.reshape(-1, C, W, H)
        images = images / 255.0 - 0.5
        with torch.no_grad():
            # Image features
            image_feat = self.visual_encoder(images)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True) 
            image_feat = image_feat.reshape(-1, self.num_times * self.channel)

            # Store for zero-shot
            self.feat_store.append(image_feat)
            self.label_store.append(label)

            # Text features
            text_feat = self.textual_encoder()
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)  

            # Classification logits
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_feat @ text_feat.t() * 1.0
        
        return logits
