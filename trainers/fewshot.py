
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from trainers.mv_utils_fs import PCViews
import torchvision.transforms as transforms






CUSTOM_TEMPLATES = {
    'NMNIST_DATA': 'A black and white photo of the Arabic numeral number: "{}".',
    'CIFAR10DVS_DATA': 'A low quality stick figure of the object {}.',
    'ESIMAGENET_DATA': 'A low quality point cloud image of {}.'
}

# source: https://github.com/WangYueFt/dgcnn/blob/master/pytorch/util.py
def smooth_loss(pred, gold):
    eps = 0.2
    n_class = pred.size(1)

    one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)

    loss = -(one_hot * log_prb).sum(dim=1).mean()
    return loss

class BatchNormPoint(nn.Module):
    def __init__(self, feat_size, sync_bn=False):
        super().__init__()
        self.feat_size = feat_size
        self.sync_bn=sync_bn
        if self.sync_bn:
            self.bn = BatchNorm2dSync(feat_size)
        else:
            self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        if self.sync_bn:
            # 4d input for BatchNorm2dSync
            x = x.view(s1 * s2, self.feat_size, 1, 1)
            x = self.bn(x)
        else:
            x = x.view(s1 * s2, self.feat_size)
            x = self.bn(x)
        return x.view(s1, s2, s3)

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model

       
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


class PointCLIP_Model(nn.Module):

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        
        # Encoders from CLIP
        self.visual_encoder = clip_model.visual
        self.textual_encoder = Textual_Encoder(cfg, classnames, clip_model)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Multi-view projection
        self.num_times = cfg.MODEL.PROJECT.NUM_TIMES
        pc_views = PCViews()
        self.get_img = pc_views.get_img

        # inter-view Adapter
        self.adapter = Adapter(cfg).to(clip_model.dtype)

        # Store features for post-process view-weight search
        self.store = False
        self.feat_store = []
        self.label_store = []

    
    def forward(self, pc, label=None): 

        # Project to multi-view depth maps
        images = self.mv_proj(pc).type(self.dtype)
        B,T,C,W,H = images.shape
        images = images.reshape(-1, C, W, H)
        images = images / 255.0 - 0.5
        
        # Image features
        image_feat = self.visual_encoder(images)
        image_feat = self.adapter(image_feat)
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)   

        # Store for the best ckpt
        if self.store:
            self.feat_store.append(image_feat)
            self.label_store.append(label)

        # Text features
        text_feat = self.textual_encoder()
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        # Classification logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_feat @ text_feat.t() * 1.

        return logits

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
                    tmp_i = torch.where(tmp > 0)   #for cifar10-dvs, please change to tmp > 1
                    output_img[:,j,:,:][tmp_i] = 255
                else:
                    tmp_i = torch.where(tmp > 0)   #for cifar10-dvs, please change to tmp > 1
                    output_img[:,j,:,:][tmp_i] = 0
                #output_img += tmp_i.unsqueeze(1).repeat(1, 3, 1, 1)
                
                #output_img += tmp_i.unsqueeze(1).repeat(1, 3, 1, 1)  
        #output_img = torch.nn.functional.upsample(output_img, size=(224, 224), mode='nearest')
        #output_img = torch.nn.functional.upsample(output_img, size=(48, 48), mode='bilinear')
        output_img = torch.nn.functional.upsample(output_img, size=(224, 224), mode='bilinear')
        output_img = output_img.unsqueeze(2).repeat(1, 1, 3, 1, 1)     
        return output_img




def fire_function(gamma):
    class ZIF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            out = (input >= 0).half()
            ctx.save_for_backward(input)
            return out

        @staticmethod
        def backward(ctx, grad_output):
            (input, ) = ctx.saved_tensors
            grad_input = grad_output.clone()
            tmp = (input.abs() < gamma/2).half() / gamma
            grad_input = grad_input * tmp
            return grad_input, None

    return ZIF.apply


class LIFSpike(nn.Module):
    def __init__(self, thresh=0.5, tau=0.25, gamma=1.0):
        super(LIFSpike, self).__init__()
        self.thresh = thresh
        self.tau = tau
        self.gamma = gamma
        self.mem = 0

    def forward(self, x):
        self.mem = self.mem * self.tau + x
        spike = fire_function(self.gamma)(self.mem - self.thresh)
        self.mem = (1 - spike) * self.mem
        return spike




class Adapter(nn.Module):
    """
    Inter-view Adapter
    """

    def __init__(self, cfg):
        super().__init__()

        self.num_times = cfg.MODEL.PROJECT.NUM_TIMES
        self.in_features = cfg.MODEL.BACKBONE.CHANNEL
        self.adapter_ratio = cfg.MODEL.ADAPTER.RATIO
        self.fusion_init = cfg.MODEL.ADAPTER.INIT
        self.dropout = cfg.MODEL.ADAPTER.DROPOUT

        
        #self.fusion_ratio = nn.Parameter(torch.tensor([self.fusion_init] * self.num_times), requires_grad=True)
        
        self.global_f = nn.Sequential(
                nn.BatchNorm1d(self.in_features),
                nn.Dropout(self.dropout),
                nn.Linear(in_features=self.in_features,
                          out_features=int(self.in_features/8)),
                nn.BatchNorm1d(int(self.in_features/8)),
                LIFSpike(),
                nn.Dropout(self.dropout))

        self.view_f = nn.Sequential(
                #nn.Linear(in_features=int(self.in_features/8),
                #          out_features=int(self.in_features/8)),
                #LIFSpike(),
                nn.Linear(in_features=int(self.in_features/8),
                          out_features=self.in_features),
                nn.ReLU())


    def forward(self, feat):
        for m in self.modules():
            if isinstance(m, LIFSpike):
                m.mem = 0
        img_feat = feat.reshape(-1, self.num_times, self.in_features)
        img_feat = img_feat.permute([1, 0, 2])
        res_feat = feat.reshape(-1, self.num_times * self.in_features)
        
        all_outputs = []
        for i in range(self.num_times):
            # Global feature
            global_feat = self.global_f(img_feat[i])
            # View-wise adapted features
            view_feat = self.view_f(global_feat)
            
            all_outputs += [view_feat]
        out = torch.stack(all_outputs)
        out = out.permute([1, 0, 2])
        out = out.reshape(-1, self.num_times * self.in_features)
        img_feat = out * self.adapter_ratio + res_feat * (1 - self.adapter_ratio)

        return img_feat

flip = transforms.RandomHorizontalFlip()
rotate = transforms.RandomRotation(degrees=10)
shearx = transforms.RandomAffine(degrees=0, shear=(-5, 5))
@TRAINER_REGISTRY.register()
class NeuroCLIP_FS(TrainerX):
    """
        PointCLIP: Point Cloud Understanding by CLIP
        https://arxiv.org/pdf/2112.02413.pdf
    """ 

        
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)

        print('Building PointCLIP')
        self.model = PointCLIP_Model(cfg, classnames, clip_model)

        print('Turning off gradients in both visual and textual encoders')
        for name, param in self.model.named_parameters():
            if 'adapter' not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        
        self.model.to(self.device)
        self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model('adapter', self.model.adapter, self.optim, self.sched)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        B, T, C, W, H = image.shape
        all_images = []
        
        for i in range(B):
            oneimage = rotate(image[i])
            oneimage = flip(oneimage)
            oneimage = shearx(oneimage)
            all_images += [oneimage]
        image = torch.stack(all_images)
        
        
        output = self.model(image)
        loss = smooth_loss(output, label)
        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )

            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
