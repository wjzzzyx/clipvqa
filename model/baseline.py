import os
import clip
from collections import OrderedDict
import torch
import torch.nn as nn
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

        self.attn = nn.MultiheadAttention(d_model, n_head)
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


class ClipTransEncoder(nn.Module):

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
        
        else:
            raise NotImplementedError('Unsupported clip type.')

        self.visual_prompt = nn.Parameter(torch.zeros(cfg.TRAIN.VISION.PREFIX_LEN, cfg.TRAIN.VISION.HID_DIM))
        self.question_prompt = nn.Parameter(torch.zeros(cfg.TRAIN.QUESTION.PREFIX_LEN, cfg.TRAIN.QUESTION.HID_DIM))
        self.answer_prompt = nn.Parameter(torch.zeros(cfg.TRAIN.ANSWER.PREFIX_LEN, cfg.TRAIN.ANSWER.HID_DIM))
        self.cls_emb = nn.Parameter(torch.zeros(1, cfg.TRAIN.VISION.HID_DIM))
        self.sep_emb = nn.Parameter(torch.zeros(1, cfg.TRAIN.VISION.HID_DIM))
        nn.init.normal_(self.visual_prompt, mean=0.0, std=0.01)
        nn.init.normal_(self.question_prompt, mean=0.0, std=0.01)
        nn.init.normal_(self.answer_prompt, mean=0.0, std=0.01)
        nn.init.normal_(self.cls_emb, mean=0.0, std=0.01)
        nn.init.normal_(self.sep_emb, mean=0.0, std=0.01)

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
    
    def get_origin_clip_visual_embeddings_with_prompt(self, image):
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
        x = self.visual_projection(x)
        prompt = self.visual_prompt.expand(x.size(0), -1, -1)
        x = torch.cat((prompt, x), dim=1)
        return x

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
    
    def get_wx_clip_visual_embeddings_with_prompt(self, image):
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
        x = self.visual_projection(x)
        prompt = self.visual_prompt.expand(x.size(0), -1, -1)
        x = torch.cat((prompt, x), dim=1)
        return x

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
        input_ids = torch.cat((input_ids.new_zeros(input_ids.size(0), prompt.size(1)), input_ids), dim=1)
        attention_mask = torch.cat((attention_mask.new_ones(attention_mask.size(0), prompt.size(1)), attention_mask), dim=1)
        # if self.cfg.TRAIN.CLIP_TYPE == 'origin':
        #     x = self.text_encoder(input_ids=input_ids, inputs_embeds=embeds, attention_mask=attention_mask)
        # else:
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

        # clip visual and text encoder
        if self.cfg.TRAIN.CLIP_TYPE == 'origin':
            v_emb = self.get_origin_clip_visual_embeddings_with_prompt(v[2])
            q_emb = self.get_origin_clip_text_embeddings_with_prompt(
                question, self.question_prompt, self.cfg.TRAIN.QUESTION.LENGTH)['pooler_output'].unsqueeze(1)
        else:
            v_emb = self.get_wx_clip_visual_embeddings_with_prompt(v[2])
            q_emb = self.get_wx_clip_text_embeddings_with_prompt(
                question, self.question_prompt)['pooler_output'].unsqueeze(1)

        # get open & close feature
        # separate on batch dim
        v_open, v_close, q_open, q_close, a_open, a_close, _, _ \
            = seperate(v_emb, q_emb, a, is_open, self.dataset.num_close_candidates)

        # cls_emb = self.cls_emb.expand(v_open.size(0), -1, -1)
        sep_emb = self.sep_emb.expand(v_open.size(0), -1, -1)
        multimodal_open_input = torch.cat((q_open, sep_emb, v_open), dim=1)
        multimodal_open_output = self.multimodal_encoder_open(multimodal_open_input)
        # cls_emb = self.cls_emb.expand(v_close.size(0), -1, -1)
        sep_emb = self.sep_emb.expand(v_close.size(0), -1, -1)
        multimodal_close_input = torch.cat((q_close, sep_emb, v_close), dim=1)
        multimodal_close_output = self.multimodal_encoder_close(multimodal_close_input)
        mm_open_emb = multimodal_open_output[:, 0]
        mm_close_emb = multimodal_close_output[:, 0]

        self.open_embeds, self.close_embeds = self.embed_all_answers()

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
    

    def embed_all_answers(self):
        answers = [k for k, v in sorted(self.dataset.ans2label.items(), key=lambda item: item[1])]
        open_answers = self.dataset.label2open
        close_answers = self.dataset.label2close
        if self.cfg.TRAIN.CLIP_TYPE == 'origin':
            # self.answer_embeds = self.get_origin_clip_embeddings_with_prompt(answers)['pooler_output']
            open_embeds = self.get_origin_clip_text_embeddings_with_prompt(
                open_answers, self.answer_prompt, self.cfg.TRAIN.QUESTION.LENGTH)['pooler_output']
            close_embeds = self.get_origin_clip_text_embeddings_with_prompt(
                close_answers, self.answer_prompt, self.cfg.TRAIN.QUESTION.LENGTH)['pooler_output']
        else:
            # self.answer_embeds = self.get_wx_clip_embeddings_with_prompt(answers)['pooler_output']
            open_embeds = self.get_wx_clip_text_embeddings_with_prompt(
                open_answers, self.answer_prompt)['pooler_output']
            close_embeds = self.get_wx_clip_text_embeddings_with_prompt(
                close_answers, self.answer_prompt)['pooler_output']
        return open_embeds, close_embeds
        