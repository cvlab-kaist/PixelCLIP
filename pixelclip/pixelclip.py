from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_sem_seg_head
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList, Instances

import copy
from einops import rearrange
from fast_pytorch_kmeans import KMeans

@META_ARCH_REGISTRY.register()
class PixelCLIP(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        sem_seg_head: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        clip_finetune: str,
        eps: float,
        dino: bool,
        max_masks: int,
        clip_resolution: Tuple[int],
        mask_resolution: Tuple[int],
    ):
        super().__init__()
        self.sem_seg_head = sem_seg_head

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)

        self.clip_finetune = clip_finetune

        for name, params in self.sem_seg_head.named_parameters():
            params.requires_grad = False if "prompt" not in name else True
            
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "visual" in name:
                if clip_finetune == "conv":
                    params.requires_grad = True if "conv" in name else False
                elif clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    if "attn" in name:
                        params.requires_grad = True 
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False
                
        self.teacher_encoder = copy.deepcopy(self.sem_seg_head.predictor.clip_model)
        self.teacher_encoder.transformer = None # remove text encoder
        
        for name, params in self.teacher_encoder.named_parameters():
            params.requires_grad = False
            
        self.max_masks = max_masks
        self.teacher_res = clip_resolution
        self.student_res = self.teacher_res
        self.mask_res = mask_resolution
        self.head = ConvHead()
        
        self.cluster_embeddings = None
        self.eps = eps 
        self.dino = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(self.device) if dino else None
        if dino:
            self.patch_size = 16
            self.dino_cluster = 16
            for name, params in self.dino.named_parameters():
                params.requires_grad = False
        

    @classmethod
    def from_config(cls, cfg):
        sem_seg_head = build_sem_seg_head(cfg, None)
        
        return {
            "sem_seg_head": sem_seg_head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "eps": cfg.MODEL.EPS,
            "dino": cfg.MODEL.DINO,
            "max_masks": cfg.MODEL.MAX_MASKS,
            "clip_resolution": cfg.MODEL.CLIP_RESOLUTION,
            "mask_resolution": cfg.MODEL.MASK_RESOLUTION,
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    def momentum_update(self, student, teacher, alpha=0.999):
        for param_s, param_t in zip(student.parameters(), teacher.parameters()):
            param_t.data.mul_(alpha).add_(1 - alpha, param_s.data)   
    
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        
        images = [x["image"].to(self.device) for x in batched_inputs]

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images) #, self.size_divisibility)
        student_images = clip_images.tensor
        
        if "image_strong" in batched_inputs[0]:
            images_strong = [x["image_strong"].to(self.device) for x in batched_inputs]
            images_strong = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images_strong]
            images_strong = ImageList.from_tensors(images_strong)
            student_images = images_strong.tensor
        
        clip_images_student = F.interpolate(student_images, size=self.student_res, mode='bilinear', align_corners=False, )
        clip_feat = self.sem_seg_head(clip_images_student, dense=True)
        clip_feat_normalized = F.normalize(clip_feat, dim=-1)

        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            elif self.dino is not None:
                gt_instances = self.get_dino_masks(clip_images)
            
            with torch.no_grad():
                self.momentum_update(self.sem_seg_head.predictor.clip_model.visual, self.teacher_encoder.visual)
                
                clip_images_teacher = F.interpolate(clip_images.tensor, size=self.teacher_res, mode='bilinear', align_corners=False, )
                teacher_feat = self.teacher_encoder.encode_image(clip_images_teacher, dense=True).permute(0, 3, 1, 2)
            
            self.cluster_embeddings = self.sem_seg_head.predictor.get_prompted_text_embeds()
            clustered_targets = self.prepare_clusters(gt_instances, teacher_feat)
            cost_volume = torch.einsum("nc, bhwc -> bnhw", F.normalize(self.cluster_embeddings, dim=-1), clip_feat_normalized).reshape(-1, *clip_feat_normalized.shape[1:3])
            
            valid_idx = clustered_targets.sum(dim=[1]) > 0
            
            mask_preds = self.head(cost_volume.unsqueeze(1))
            mask_preds = F.interpolate(mask_preds, size=self.mask_res, mode="bilinear").squeeze().reshape(-1, self.cluster_embeddings.shape[0], *self.mask_res)
            loss = F.binary_cross_entropy_with_logits(mask_preds.permute(0, 2, 3, 1)[valid_idx], clustered_targets.float().permute(0, 2, 3, 1)[valid_idx])
            
            return {"loss_masks": loss, } 
            
        else:
            clip_feat = F.interpolate(clip_feat.permute(0, 3, 1, 2), size=(64, 64), mode="bilinear").permute(0, 2, 3, 1)
            
            text_feat = self.sem_seg_head.predictor.text_features_test
            if text_feat.dim() == 3:
                text_feat = text_feat.mean(dim=1)
            text_feat = F.normalize(text_feat, dim=-1)
            clip_feat = F.normalize(clip_feat, dim=-1)            
            
            outputs = clip_feat @ text_feat.T
            outputs = outputs.permute(0, 3, 1, 2)

            image_size = clip_images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results

    # Modified by Heeseong Shin from SwAV: https://github.com/facebookresearch/swav/blob/06b1b7cbaf6ba2a792300d79c7299db98b93b7f9/main_swav.py#L354
    @torch.no_grad()
    def distributed_sinkhorn(self, logits, n_iters=3):
        Q = logits.softmax(dim=-1).T 
        B = logits.shape[1] # number of samples to assign
        K = logits.shape[0] # how many prototypes
        
        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
            
        B = sum_Q
        Q /= sum_Q

        for it in range(n_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B # the colomns must sum to 1 so that Q is an assignment
        
        return Q.t()
    
    def get_dino_masks(self, images):
        dino_images = ((images.tensor * self.clip_pixel_std) + self.clip_pixel_mean - self.pixel_mean) / self.pixel_std
        dino_images = F.interpolate(dino_images, size=(dino_images.shape[-2] // 16 * self.patch_size, dino_images.shape[-1] // 16 * self.patch_size), mode="bilinear", align_corners=False)
        
        dino_features = self.dino.get_intermediate_layers(dino_images)[0][:, 1:]
        n_clusters = 16
        kmeans = KMeans(n_clusters=n_clusters, )
        clusters = torch.stack([kmeans.fit_predict(dino_feature) for dino_feature in dino_features], dim=0)
        h, w = dino_images.shape[-2]//self.patch_size, dino_images.shape[-1]//self.patch_size
        
        dino_masks = torch.zeros((len(images), n_clusters, clusters.shape[-1]), dtype=torch.bool, device=images.tensor.device)
        dino_masks = dino_masks.scatter_(1, clusters.unsqueeze(1), 1).reshape(len(images), n_clusters, h, w)

        dino_masks_resized = F.interpolate(dino_masks.float(), size=images.tensor.shape[-2:], mode="nearest").to(bool)
        
        return dino_masks_resized
    
    def prepare_clusters(self, targets, teacher_feats):
        teacher_feats = F.interpolate(teacher_feats, size=self.mask_res, mode="bilinear").permute(0, 2, 3, 1)
        target_label = torch.zeros(teacher_feats.shape[0], self.cluster_embeddings.shape[0], self.mask_res[0], self.mask_res[1], device=teacher_feats.device)
        num_masks = torch.zeros(teacher_feats.shape[0], dtype=torch.long, device=teacher_feats.device)
        
        logits = []
        down_masks = []
        
        for i, targets_per_image in enumerate(targets):
            if isinstance(targets_per_image, torch.Tensor):
                gt_masks = targets_per_image
            else:
                gt_masks = targets_per_image.gt_masks.tensor
            if gt_masks.shape[0] == 0:
                continue
            
            # downsize masks for masked pooling
            downsized_masks = F.interpolate(gt_masks.float().unsqueeze(0), size=self.mask_res, mode="nearest").squeeze(0)
            filter_idx = downsized_masks.sum(dim=[1, 2]) > 1 # filter out small masks lost in downsizing
            downsized_masks = downsized_masks[filter_idx]
            
            if downsized_masks.shape[0] == 0:
                continue
            
            # compute mask features
            mask_feats = torch.einsum("hwc, nhw -> nc", teacher_feats[i], downsized_masks / downsized_masks.sum(dim=[1, 2], keepdim=True))
            similarity_map = F.normalize(mask_feats, dim=-1) @ F.normalize(self.cluster_embeddings, dim=-1).T # image-cluster similarity map
            
            logit = (self.eps * similarity_map) 
            logits.append(logit)
            down_masks.append(downsized_masks)
            
            num_masks[i] = downsized_masks.shape[0]
        
        logits = torch.cat(logits, dim=0)
        assignment = self.distributed_sinkhorn(logits)
        hard_assignment = assignment.argmax(dim=-1)
        
        down_masks = torch.cat(down_masks, dim=0)
        
        # merge GT masks with respect to assigned clusters
        for i, n_mask in enumerate(num_masks):
            assign = hard_assignment[num_masks[:i].sum():num_masks[:i].sum() + num_masks[i]]
            mask = down_masks[num_masks[:i].sum():num_masks[:i].sum() + num_masks[i]]
            target_label[i, assign] += mask
            target_label[i].index_add_(0, assign, mask)

        target_label = target_label.clamp(0, 1)
        return target_label


class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class ConvHead(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, mid_channels=[64, 64, 32], guidance_channels=None):
        super().__init__()
        if guidance_channels is None:
            guidance_channels = [0, 0, 0]
        self.conv0 = nn.Conv2d(in_channels, mid_channels[0], kernel_size=3, padding=1)
        
        self.up1 = nn.ConvTranspose2d(mid_channels[0], mid_channels[1], kernel_size=2, stride=2)
        self.conv1 = DoubleConv(mid_channels[1] + guidance_channels[1], mid_channels[1] + guidance_channels[1])
        
        self.up2 = nn.ConvTranspose2d(mid_channels[1], mid_channels[2], kernel_size=2, stride=2)
        self.conv2 = DoubleConv(mid_channels[2] + guidance_channels[2], mid_channels[2], guidance_channels[2])
        
        self.proj = nn.Conv2d(mid_channels[2], out_channels, kernel_size=3, padding=1)
        
    def forward(self, x, guidance=None):
        x = self.conv0(x)
        x = self.conv1(self.up1(x))
        x = self.conv2(self.up2(x))
        return self.proj(x)