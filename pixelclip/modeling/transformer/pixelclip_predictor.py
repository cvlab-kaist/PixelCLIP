import torch

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from pixelclip.third_party import clip
from pixelclip.third_party import imagenet_templates

import numpy as np
import open_clip
from open_clip.transformer import text_global_pool

class PixelCLIPPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        train_class_json: str,
        test_class_json: str,
        clip_pretrained: str,
        prompt_ensemble_type: str,
        prompt_length: int,
        num_classes: int,
    ):
        """
        Args:
            
        """
        super().__init__()
        
        import json
        # use class_texts in train_forward, and test_class_texts in test_forward
        with open(train_class_json, 'r') as f_in:
            self.class_texts = json.load(f_in)
        with open(test_class_json, 'r') as f_in:
            self.test_class_texts = json.load(f_in)
        assert self.class_texts != None
        if self.test_class_texts == None:
            self.test_class_texts = self.class_texts
        device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.tokenizer = None
        if "convnext" in clip_pretrained:
            name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup') if "base" not in clip_pretrained else ('convnext_base_w_320', 'laion_aesthetic_s13b_b82k')
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrain, device=device)
            self.tokenizer = open_clip.get_tokenizer(name)
            
            def custom_image(self, image, dense=False, normalize=False):
                if dense:
                    features = self.visual.trunk.forward_features(image)
                    features = self.visual.trunk.head.norm(features) # skip global pooling
                    features = self.visual.head(features.permute(0, 2, 3, 1))
                else:
                    features = self.visual(image)
                return F.normalize(features, dim=-1) if normalize else features
            
            funcType = type(clip_model.encode_image)
            clip_model.encode_image = funcType(custom_image, clip_model)
            
            def custom_text(self, text, normalize: bool = False, prompt = None):
                cast_dtype = self.transformer.get_cast_dtype()

                if prompt is not None:
                    x = prompt
                else:
                    x = self.token_embedding(text).to(cast_dtype)   # [batch_size, n_ctx, d_model]

                x = x + self.positional_embedding.to(cast_dtype)
                x = self.transformer(x, attn_mask=self.attn_mask)
                x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
                x, _ = text_global_pool(x, text, self.text_pool_type)
                if self.text_projection is not None:
                    if isinstance(self.text_projection, nn.Linear):
                        x = self.text_projection(x)
                    else:
                        x = x @ self.text_projection

                return F.normalize(x, dim=-1) if normalize else x
            
            funcType = type(clip_model.encode_text)
            clip_model.encode_text = funcType(custom_text, clip_model)
        else:
            # for OpenAI models
            clip_model, clip_preprocess = clip.load(clip_pretrained, device=device, jit=False)
            self.tokenizer = clip.tokenize
    
        self.prompt_ensemble_type = prompt_ensemble_type        

        if self.prompt_ensemble_type == "imagenet_select":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES_SELECT
        elif self.prompt_ensemble_type == "imagenet":
            prompt_templates = imagenet_templates.IMAGENET_TEMPLATES
        elif self.prompt_ensemble_type == "single":
            prompt_templates = ['A photo of a {} in the scene',]
        elif self.prompt_ensemble_type == "vild":
            prompt_templates = imagenet_templates.VILD
        else:
            raise NotImplementedError
        
        self.prompt_templates = prompt_templates

        self.text_features = self.class_embeddings(self.class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        self.text_features_test = self.class_embeddings(self.test_class_texts, prompt_templates, clip_model).permute(1, 0, 2).float()
        
        torch.manual_seed(0)
        self.num_classes = num_classes
        self.length = prompt_length
        
        with torch.no_grad():
            prompt_words = torch.randint(clip_model.vocab_size - 10, (self.num_classes, self.length), dtype=torch.long).cuda()

            self.template_token = self.tokenizer(['A photo of a {}in the scene'.format('{} ' * self.length)]).cuda()
            self.token_pos = 39025 # tokenizer index of '{}'
            
            if not hasattr(clip_model, 'token_embedding'):
                self.prompt_template = clip_model.text.token_embedding(self.template_token)
                prompt_tokens = clip_model.text.token_embedding(prompt_words)
            else:
                self.prompt_template = clip_model.token_embedding(self.template_token)
                prompt_tokens = clip_model.token_embedding(prompt_words)
        
            self.ensemble = self.prompt_ensemble_type != "single"
            # for ensemble during training
            if self.ensemble:
                self.ensemble_prompt_template = []
                self.ensemble_template_token = []
                for prompt in self.prompt_templates:
                    _template_token = self.tokenizer([prompt.format('{} ' * self.length, article="a")]).cuda()
                    
                    if not hasattr(clip_model, 'token_embedding'):
                        _prompt_template = clip_model.text.token_embedding(self.template_token)
                    else:
                        _prompt_template = clip_model.token_embedding(self.template_token)
                    
                    self.ensemble_prompt_template.append(_prompt_template)
                    self.ensemble_template_token.append(_template_token)

        self.prompt_tokens = nn.Parameter(prompt_tokens, requires_grad=True)
        
        self.clip_model = clip_model.float()
        self.clip_preprocess = clip_preprocess
        
        self.tokens = None
        self.cache = None

    @classmethod
    def from_config(cls, cfg):
        ret = {}

        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON
        ret["clip_pretrained"] = cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED
        ret["prompt_ensemble_type"] = cfg.MODEL.PROMPT_ENSEMBLE_TYPE

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH
        return ret

    def forward(self, x, dense=True):
        return self.clip_model.encode_image(x, dense=dense)

    @torch.no_grad()
    def class_embeddings(self, classnames, templates, clip_model):
        def article(name):
            return 'an' if name[0] in 'aeiou' else 'a'
        
        zeroshot_weights = []
        for classname in classnames:
            if ', ' in classname:
                classname_splits = classname.split(', ')
                texts = []
                for template in templates:
                    for cls_split in classname_splits:
                        texts.append(template.format(cls_split, article=article(cls_split)))
            else:
                texts = [template.format(classname, article=article(classname)) for template in templates]  # format with class
            if self.tokenizer is not None:
                texts = self.tokenizer(texts).cuda()
            else: 
                texts = clip.tokenize(texts).cuda()

            class_embeddings = clip_model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            if len(templates) != class_embeddings.shape[0]:
                class_embeddings = class_embeddings.reshape(len(templates), -1, class_embeddings.shape[-1]).mean(dim=1)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights
    
    def get_prompted_text_embeds(self, ensemble=False):
        if self.ensemble:
            return self.get_prompted_text_embeds_ensemble()
        else:
            return self._get_prompted_text_embeds()
    
    def _get_prompted_text_embeds(self, ):
        normalize = False
        if self.prompt_tokens.dim() == 2:
            return self.prompt_tokens

        text = self.template_token.repeat(self.num_classes, 1)
        mask = text == self.token_pos
        prompt = self.prompt_template.repeat(self.num_classes, 1, 1)
        if normalize:
            _tokens = self.prompt_tokens.reshape(-1, self.prompt_tokens.shape[-1]) - self.prompt_tokens.mean(dim=(0,1))
            _tokens = _tokens / self.prompt_tokens.std(dim=(0,1))
            _tokens = _tokens * self.prompt_std + self.prompt_mean
            prompt[mask] = _tokens
        else:
            prompt[mask] = self.prompt_tokens.reshape(-1, self.prompt_tokens.shape[-1])
        text_features = self.clip_model.encode_text(text, prompt=prompt)
        return text_features
    
    def get_prompted_text_embeds_ensemble(self, ):
        text_embeds = []

        for prompt_template, template_token in zip(self.ensemble_prompt_template, self.ensemble_template_token):
            text = template_token.repeat(self.num_classes, 1)
            mask = text == self.token_pos
            prompt = prompt_template.repeat(self.num_classes, 1, 1)
            prompt[mask] = self.prompt_tokens.reshape(-1, self.prompt_tokens.shape[-1])
            text_features = self.clip_model.encode_text(text, prompt=prompt)
            text_embeds.append(text_features)
        
        text_features = torch.stack(text_embeds, dim=0).mean(dim=0)
        
        return text_features
    
    def get_text_embeds(self, classnames, templates, clip_model, prompt=None):
        if self.cache is not None and not self.training and prompt is None:
            return self.cache
        
        if self.tokens is None or prompt is not None:
            def article(name):
                return 'an' if name[0] in 'aeiou' else 'a'
            tokens = []
            for classname in classnames:
                if ', ' in classname:
                    classname_splits = classname.split(', ')
                    texts = [template.format(classname_splits[0], article=article(classname_splits[0])) for template in templates]
                else:
                    texts = [template.format(classname, article=article(classname)) for template in templates]  # format with class
                if self.tokenizer is not None:
                    texts = self.tokenizer(texts).cuda()
                else: 
                    texts = clip.tokenize(texts).cuda()
                tokens.append(texts)
            tokens = torch.stack(tokens, dim=0).squeeze(1)
            if prompt is None:
                self.tokens = tokens
        elif self.tokens is not None and prompt is None:
            tokens = self.tokens

        class_embeddings = clip_model.encode_text(tokens, prompt)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        
        class_embeddings = class_embeddings.unsqueeze(1)
        
        if not self.training:
            self.cache = class_embeddings
            
        return class_embeddings