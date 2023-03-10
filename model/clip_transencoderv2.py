import os
import clip
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from transformers import AutoTokenizer, AutoModel


def seperate(v, q, a, answer_target, n_unique_close):
    indexs_open = []
    indexs_close = []

    for i in range(len(answer_target)):
        if answer_target[i]==0:
            indexs_close.append(i)
        else:
            indexs_open.append(i)
        if len(q.shape) == 2:  # in case of using clip to encode q
            q = q.unsqueeze(1)
    return v[indexs_open,:,:],v[indexs_close,:,:],\
        q[indexs_open,:,:],q[indexs_close,:,:],\
        a[indexs_open, n_unique_close:],a[indexs_close,:n_unique_close],\
        indexs_open, indexs_close


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        if attn_mask is None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        for block in self.resblocks:
            x = block(x, attn_mask)
        return x


class ClipTransEncoderV2(nn.Module):

    def __init__(self, dataset, cfg):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset

        # build and load pre-trained Auto-encoder model
        # if cfg.TRAIN.VISION.AUTOENCODER:
        #     self.ae = Auto_Encoder_Model()
        #     weight_path = cfg.DATASET.DATA_DIR + '/' + cfg.TRAIN.VISION.AE_PATH
        #     print('load initial weights DAE from: %s' % (weight_path))
        #     self.ae.load_state_dict(torch.load(weight_path))
        #     self.convert = nn.Linear(16384, 64)
        
        # build and load pre-trained CLIP model
        if cfg.TRAIN.CLIP_TYPE == 'origin':
            # original clip
            self.clip, _ = clip.load('RN50')
            self.clip = self.clip.to(torch.float32)
            for param in self.clip.parameters():
                param.requires_grad_(False)
            self.visual_projection = nn.Linear(cfg.TRAIN.VISION.V_DIM, cfg.TRAIN.VISION.HID_DIM, bias=False)
            self.visual_projection2 = nn.Linear(cfg.TRAIN.VISION.POOL_DIM, cfg.TRAIN.MULTIMODAL.WIDTH, bias=False)
            self.text_projection = nn.Parameter(torch.empty(cfg.TRAIN.QUESTION.HID_DIM, cfg.TRAIN.QUESTION.HID_DIM))
            torch.nn.init.normal_(self.text_projection, std=cfg.TRAIN.QUESTION.HID_DIM ** -0.5)
        
        elif cfg.TRAIN.CLIP_TYPE == 'text_only':
            self.image_encoder = torch.load(cfg.TRAIN.IMAGE_ENCODER_PATH)
            self.text_encoder = AutoModel.from_pretrained(cfg.TRAIN.TEXT_ENCODER_PATH)
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.TRAIN.TEXT_ENCODER_PATH)
            for param in self.text_encoder.parameters():
                param.requires_grad_(False)
            for param in self.image_encoder.parameters():
                param.requires_grad_(False)
        
        elif cfg.TRAIN.CLIP_TYPE == 'weixiong':
            self.image_encoder = torch.load(cfg.TRAIN.IMAGE_ENCODER_PATH, map_location='cpu')
            self.text_encoder = torch.load(cfg.TRAIN.TEXT_ENCODER_PATH, map_location='cpu')
            self.tokenizer = self.text_encoder.tokenizer
            self.text_encoder = self.text_encoder.text_encoder
            for param in self.text_encoder.parameters():
                param.requires_grad_(False)
            for param in self.image_encoder.parameters():
                param.requires_grad_(False)
            self.visual_projection = nn.Linear(cfg.TRAIN.VISION.V_DIM, cfg.TRAIN.VISION.HID_DIM, bias=False)
            self.visual_projection2 = nn.Linear(cfg.TRAIN.VISION.POOL_DIM, cfg.TRAIN.MULTIMODAL.WIDTH, bias=False)
        
        elif cfg.TRAIN.CLIP_TYPE == 'weixiongv2':
            self.image_encoder = torch.load(cfg.TRAIN.IMAGE_ENCODER_PATH, map_location='cpu')
            self.text_encoder = torch.load(cfg.TRAIN.TEXT_ENCODER_PATH, map_location='cpu')
            self.tokenizer = AutoTokenizer.from_pretrained(cfg.TRAIN.TOKENIZER_PATH)
            for param in self.text_encoder.parameters():
                param.requires_grad_(False)
            for param in self.image_encoder.parameters():
                param.requires_grad_(False)
            self.visual_projection = nn.Linear(cfg.TRAIN.VISION.V_DIM, cfg.TRAIN.VISION.HID_DIM, bias=False)

        else:
            raise NotImplementedError('Unsupported clip type.')
        
        self.question_prompt = nn.Parameter(torch.zeros(cfg.TRAIN.QUESTION.PREFIX_LEN, cfg.TRAIN.QUESTION.HID_DIM))
        nn.init.normal_(self.question_prompt, mean=0.0, std=0.01)
        self.answer_prompt = nn.Parameter(torch.zeros(cfg.TRAIN.ANSWER.PREFIX_LEN, cfg.TRAIN.ANSWER.HID_DIM))
        nn.init.normal_(self.answer_prompt, mean=0.0, std=0.01)
        self.sep_emb = nn.Parameter(torch.zeros(1, cfg.TRAIN.VISION.HID_DIM))
        nn.init.normal_(self.sep_emb, mean=0.0, std=0.01)
        self.positional_embedding = nn.Parameter(torch.empty(50 + 1 + cfg.TRAIN.QUESTION.PREFIX_LEN + cfg.TRAIN.QUESTION.LENGTH, cfg.TRAIN.QUESTION.HID_DIM))
        nn.init.normal_(self.positional_embedding, std=0.01)

        self.multimodal_encoder_open = Transformer(
            cfg.TRAIN.MULTIMODAL.WIDTH, cfg.TRAIN.MULTIMODAL.LAYERS, cfg.TRAIN.MULTIMODAL.HEADS)
        self.multimodal_encoder_close = Transformer(
            cfg.TRAIN.MULTIMODAL.WIDTH, cfg.TRAIN.MULTIMODAL.LAYERS, cfg.TRAIN.MULTIMODAL.HEADS)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask
    
    def get_origin_clip_visual_embeddings_with_prompt(self, image, prompt):
        x = self.clip.visual.relu1(self.clip.visual.bn1(self.clip.visual.conv1(image)))
        x = self.clip.visual.relu2(self.clip.visual.bn2(self.clip.visual.conv2(x)))
        x = self.clip.visual.relu3(self.clip.visual.bn3(self.clip.visual.conv3(x)))
        x = self.clip.visual.avgpool(x)
        x = self.clip.visual.layer1(x)
        x = self.clip.visual.layer2(x)
        x = self.clip.visual.layer3(x)
        x = self.clip.visual.layer4(x)
        
        x = x.flatten(start_dim=2).permute(0, 2, 1)    # shape (B, HW, C)
        x = torch.cat((x.mean(dim=1, keepdim=True), x), dim=1)
        x = x + self.clip.visual.attnpool.positional_embedding

        h = self.visual_projection(x)
        prompt = prompt.expand(x.size(0), -1, -1)
        h = torch.cat((prompt, h), dim=1)

        x = x.permute(1, 0, 2)    # shape (HW, B, C)
        pooled, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.clip.visual.attnpool.num_heads,
            q_proj_weight=self.clip.visual.attnpool.q_proj.weight,
            k_proj_weight=self.clip.visual.attnpool.k_proj.weight,
            v_proj_weight=self.clip.visual.attnpool.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.clip.visual.attnpool.q_proj.bias, self.clip.visual.attnpool.k_proj.bias, self.clip.visual.attnpool.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.clip.visual.attnpool.c_proj.weight,
            out_proj_bias=self.clip.visual.attnpool.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.clip.visual.attnpool.training,
            need_weights=False
        )
        pooled = pooled.squeeze(0)
        pooled = self.visual_projection2(pooled)
        return {'last_hidden_state': h, 'pooler_output': pooled}

    def get_origin_clip_text_embeddings_with_prompt(self, text, prompt, context_length):
        tokens = clip.tokenize(text, context_length=context_length, truncate=True)
        tokens = tokens.to(prompt.device)
        embeds = self.clip.token_embedding(tokens)
        embeds = embeds + self.clip.positional_embedding[:embeds.size(1)]
        prompt = prompt.expand(embeds.size(0), -1, -1)
        x = torch.cat((prompt, embeds), dim=1)
        x = x.permute(1, 0, 2)
        x = self.clip.transformer(x, self.clip.build_attention_mask(x.size(0)).to(x.device))
        x = x.permute(1, 0, 2)
        x = self.clip.ln_final(x)
        pooler_output = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ self.text_projection
        return {'last_hidden_state': x, 'pooler_output': pooler_output}
    
    def get_wx_clip_visual_embeddings(self, image):
        x = self.image_encoder.relu1(self.image_encoder.bn1(self.image_encoder.conv1(image)))
        x = self.image_encoder.relu2(self.image_encoder.bn2(self.image_encoder.conv2(x)))
        x = self.image_encoder.relu3(self.image_encoder.bn3(self.image_encoder.conv3(x)))
        x = self.image_encoder.avgpool(x)
        x = self.image_encoder.layer1(x)
        x = self.image_encoder.layer2(x)
        x = self.image_encoder.layer3(x)
        x = self.image_encoder.layer4(x)

        x = x.flatten(start_dim=2).permute(0, 2, 1)    # shape (B, HW, C)
        x = torch.cat((x.mean(dim=1, keepdim=True), x), dim=1)
        x = self.visual_projection(x)
        return {'last_hidden_state': x}

    def get_wx_clip_visual_embeddings_with_prompt(self, image, prompt):
        x = self.image_encoder.relu1(self.image_encoder.bn1(self.image_encoder.conv1(image)))
        x = self.image_encoder.relu2(self.image_encoder.bn2(self.image_encoder.conv2(x)))
        x = self.image_encoder.relu3(self.image_encoder.bn3(self.image_encoder.conv3(x)))
        x = self.image_encoder.avgpool(x)
        x = self.image_encoder.layer1(x)
        x = self.image_encoder.layer2(x)
        x = self.image_encoder.layer3(x)
        x = self.image_encoder.layer4(x)
        
        x = x.flatten(start_dim=2).permute(0, 2, 1)    # shape (B, HW, C)
        x = torch.cat((x.mean(dim=1, keepdim=True), x), dim=1)
        x = x + self.image_encoder.attnpool.positional_embedding

        h = self.visual_projection(x)
        prompt = prompt.expand(x.size(0), -1, -1)
        h = torch.cat((prompt, h), dim=1)

        x = x.permute(1, 0, 2)    # shape (HW, B, C)
        pooled, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.image_encoder.attnpool.num_heads,
            q_proj_weight=self.image_encoder.attnpool.q_proj.weight,
            k_proj_weight=self.image_encoder.attnpool.k_proj.weight,
            v_proj_weight=self.image_encoder.attnpool.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.image_encoder.attnpool.q_proj.bias, self.image_encoder.attnpool.k_proj.bias, self.image_encoder.attnpool.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.image_encoder.attnpool.c_proj.weight,
            out_proj_bias=self.image_encoder.attnpool.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.image_encoder.attnpool.training,
            need_weights=False
        )
        pooled = pooled.squeeze(0)
        pooled = self.visual_projection2(pooled)
        return {'last_hidden_state': h, 'pooler_output': pooled}

    def get_wx_clip_text_embeddings_with_prompt(self, text, prompt):
        encoded_input = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.cfg.TRAIN.QUESTION.LENGTH,
            return_tensors='pt'
        )
        input_ids = encoded_input['input_ids'].to(prompt.device)
        attention_mask = encoded_input['attention_mask'].to(prompt.device)
        embeds = self.text_encoder.get_input_embeddings()(input_ids)
        prompt = prompt.expand(embeds.size(0), -1, -1)
        embeds = torch.cat((prompt, embeds), dim=1)
        attention_mask = torch.cat((attention_mask.new_ones(attention_mask.size(0), prompt.size(1)), attention_mask), dim=1)
        x = self.text_encoder(inputs_embeds=embeds, attention_mask=attention_mask, output_attentions=False)
        return x

    def forward(self, v, q, question, a, is_open):
        # # get visual feature
        # if self.cfg.TRAIN.VISION.AUTOENCODER:
        #     encoder = self.ae.forward_pass(v[1])
        #     decoder = self.ae.reconstruct_pass(encoder)
        #     ae_v_emb = encoder.view(encoder.shape[0], -1)
        #     ae_v_emb = self.convert(ae_v_emb).unsqueeze(1)
        
        # if self.cfg.TRAIN.CLIPV2:
        #     if self.cfg.TRAIN.CLIP_TYPE == 'origin':
        #         clip_v_emb = self.clip.encode_image(v[2]).unsqueeze(1)
        #     else:
        #         clip_v_emb = self.image_encoder(v[2]).unsqueeze(1)

        # if self.cfg.TRAIN.VISION.AUTOENCODER and self.cfg.TRAIN.CLIPV2:
        #     v_emb = torch.cat((clip_v_emb, ae_v_emb), 2)
        question_prompt = self.question_prompt
        answer_prompt = self.answer_prompt

        # clip visual and text encoder
        if self.cfg.TRAIN.CLIP_TYPE == 'origin':
            clip_v_output = self.get_origin_clip_visual_embeddings_with_prompt(v[2], visual_prompt)
            clip_q_output = self.get_origin_clip_text_embeddings_with_prompt(
                question, question_prompt, self.cfg.TRAIN.QUESTION.LENGTH)
        else:
            clip_v_output = self.get_wx_clip_visual_embeddings(v[2])
            clip_q_output = self.get_wx_clip_text_embeddings_with_prompt(
                question, question_prompt)
        
        if self.cfg.TRAIN.VISION.EMBED_TYPE == 'seq':
            v_emb = clip_v_output['last_hidden_state']
        elif self.cfg.TRAIN.VISION.EMBED_TYPE == 'pool':
            v_emb = clip_v_output['pooler_output'].unsqueeze(1)
        else:
            raise ValueError(f'Unsupported TRAIN.VISION.EMBED_TYPE {self.cfg.TRAIN.VISION.EMBED_TYPE}')
        
        if self.cfg.TRAIN.QUESTION.EMBED_TYPE == 'seq':
            q_emb = clip_q_output['last_hidden_state']
        elif self.cfg.TRAIN.QUESTION.EMBED_TYPE == 'pool':
            q_emb = clip_q_output['pooler_output'].unsqueeze(1)
        else:
            raise ValueError(f'Unsupported TRAIN.QUESTION.EMBED_TYPE {self.cfg.TRAIN.QUESTION.EMBED_TYPE}')

        # get open & close feature
        # separate on batch dim
        v_open, v_close, q_open, q_close, a_open, a_close, _, _ \
            = seperate(v_emb, q_emb, a, is_open, self.dataset.num_close_candidates)

        sep_emb = self.sep_emb.expand(v_open.size(0), -1, -1)
        multimodal_open_input = torch.cat((v_open, sep_emb, q_open), dim=1)
        multimodal_open_input = multimodal_open_input + self.positional_embedding
        multimodal_open_output = self.multimodal_encoder_open(multimodal_open_input)
        sep_emb = self.sep_emb.expand(v_close.size(0), -1, -1)
        multimodal_close_input = torch.cat((v_close, sep_emb, q_close), dim=1)
        multimodal_close_input = multimodal_close_input + self.positional_embedding
        multimodal_close_output = self.multimodal_encoder_close(multimodal_close_input)
        mm_open_emb = multimodal_open_output[:, 50]
        mm_close_emb = multimodal_close_output[:, 50]

        self.open_embeds, self.close_embeds = self.embed_all_answers(answer_prompt)

        return mm_close_emb, mm_open_emb, a_close, a_open
    
    def classify(self, close_feat, open_feat):
        if self.cfg.TRAIN.QAS == 'dot':
            close_qas = torch.matmul(close_feat.unsqueeze(1), self.close_embeds.unsqueeze(0).transpose(1, 2)).squeeze(1)
            open_qas = torch.matmul(open_feat.unsqueeze(1), self.open_embeds.unsqueeze(0).transpose(1, 2)).squeeze(1)
        elif self.cfg.TRAIN.QAS == 'scaled dot':
            close_qas = torch.matmul(close_feat.unsqueeze(1), self.close_embeds.unsqueeze(0).transpose(1, 2)).squeeze(1)
            open_qas = torch.matmul(open_feat.unsqueeze(1), self.open_embeds.unsqueeze(0).transpose(1, 2)).squeeze(1)
            close_qas = close_qas * close_feat.size(1) ** -0.5
            open_qas = open_qas * open_feat.size(1) ** -0.5
        elif self.cfg.TRAIN.QAS == 'cosine':
            close_feat = close_feat / close_feat.norm(dim=-1, keepdim=True)
            close_embeds = self.close_embeds / self.close_embeds.norm(dim=-1, keepdim=True)
            open_feat = open_feat / open_feat.norm(dim=-1, keepdim=True)
            open_embeds = self.open_embeds / self.open_embeds.norm(dim=-1, keepdim=True)
            close_qas = torch.matmul(close_feat.unsqueeze(1), close_embeds.unsqueeze(0).transpose(1, 2)).squeeze(1)
            open_qas = torch.matmul(open_feat.unsqueeze(1), open_embeds.unsqueeze(0).transpose(1, 2)).squeeze(1)
        elif self.cfg.TRAIN.QAS == 'scaled cosine':
            close_feat = close_feat / close_feat.norm(dim=-1, keepdim=True)
            close_embeds = self.close_embeds / self.close_embeds.norm(dim=-1, keepdim=True)
            open_feat = open_feat / open_feat.norm(dim=-1, keepdim=True)
            open_embeds = self.open_embeds / self.open_embeds.norm(dim=-1, keepdim=True)
            close_qas = torch.matmul(close_feat.unsqueeze(1), close_embeds.unsqueeze(0).transpose(1, 2)).squeeze(1)
            open_qas = torch.matmul(open_feat.unsqueeze(1), open_embeds.unsqueeze(0).transpose(1, 2)).squeeze(1)
            close_qas = close_qas * 5
            open_qas = open_qas * 5
        
        return close_qas, open_qas
    

    def embed_all_answers(self, prompt):
        answers = [k for k, v in sorted(self.dataset.ans2label.items(), key=lambda item: item[1])]
        open_answers = self.dataset.label2open
        close_answers = self.dataset.label2close
        if self.cfg.TRAIN.CLIP_TYPE == 'origin':
            # self.answer_embeds = self.get_origin_clip_embeddings_with_prompt(answers)['pooler_output']
            open_embeds = self.get_origin_clip_text_embeddings_with_prompt(
                open_answers, prompt, self.cfg.TRAIN.QUESTION.LENGTH)['pooler_output']
            close_embeds = self.get_origin_clip_text_embeddings_with_prompt(
                close_answers, prompt, self.cfg.TRAIN.QUESTION.LENGTH)['pooler_output']
        else:
            # self.answer_embeds = self.get_wx_clip_embeddings_with_prompt(answers)['pooler_output']
            open_embeds = self.get_wx_clip_text_embeddings_with_prompt(
                open_answers, prompt)['pooler_output']
            close_embeds = self.get_wx_clip_text_embeddings_with_prompt(
                close_answers, prompt)['pooler_output']
        return open_embeds, close_embeds
        