import os
import json
from pathlib import Path

import clip
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from .llama import ModelArgs, Transformer, BERTTransformer
from .tokenizer import Tokenizer
from .utils import sample_top_p, _download
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class LLaMA_adapter(nn.Module):

    def __init__(self, llama_ckpt_dir, llama_tokenizer,
                 max_seq_len=512, max_batch_size=1,
                 clip_model='ViT-L/14@336px',
                 v_embed_dim=1024, v_depth=16,
                 v_num_heads=16, v_mlp_ratio=4.0,
                 query_len=577, query_layer=32, phase="finetune"):
        super().__init__()
        # llama configs
        # with open(os.path.join(llama_ckpt_dir, "7B/params.json"), "r") as f:
        with open("./ckpts/llama_model_weights/7B/params.json", "r") as f:
            params = json.loads(f.read())
        bias_lora = phase == "finetune"
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        ) # max_batch_size only affects inferenc

        # 1. clip and clip projector
        self.clip, self.clip_transform = clip.load(clip_model,download_root='./ckpts')

        clip_dim = self.clip.visual.proj.shape[1]
        self.clip_proj = nn.Linear(clip_dim, v_embed_dim)
        self.clip_proj_norm = nn.LayerNorm(v_embed_dim)

        self.query_len = query_len
        self.query_layer = query_layer

        # 2. visual query, blocks and projector

        visual_model_args = ModelArgs(dim=1024, n_layers=16, n_heads=8, max_seq_len=577)
        visual_model_args.vocab_size = 1024
        self.visual_blocks = BERTTransformer(visual_model_args)
        self.visual_proj = nn.Linear(v_embed_dim, model_args.dim)
        self.visual_proj_norm = nn.LayerNorm(model_args.dim)

        # 3. adapter query
        self.adapter_query = nn.Embedding(
            query_len * query_layer, model_args.dim)

        # 4. tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)

        # 5. llama 
        model_args.w_bias = bias_lora
        model_args.w_lora = bias_lora
        model_args.vocab_size = self.tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        self.llama = Transformer(model_args)
        torch.set_default_tensor_type(torch.FloatTensor)

        ckpts = ['./ckpts/llama_model_weights/7B/consolidated.00.pth']
        
        for ckpt in ckpts:
            # print('load_ckpt_path:', ckpt)
            ckpt = torch.load(ckpt, map_location='cpu')
            self.llama.load_state_dict(ckpt, strict=False)
            

        for name, param in self.named_parameters():
            param.requires_grad = False

        for name, para in self.llama.named_parameters():
            if 'norm' in name:
                para.data = para.data.float()
                para.requires_grad = True
            if 'bias' in name:
                para.data = para.data.float()
                para.requires_grad = True
            if 'lora' in name:
                para.data = para.data.float()
                para.requires_grad = True
        count = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
               count += 1
               print(f"Trainable param: {name}, {param.shape}, {param.dtype}")
        

        # 6. training criterion
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    def clip_encode_image(self, x):
        
        # modified from CLIP
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                      x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x

    def forward_visual(self, imgs):
        clip_feats = self.clip_encode_image(imgs)
        clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))

        visual_query = clip_feats
        visual_query = self.visual_blocks(visual_query, 0)

        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)

        return visual_query

    def forward(self, tokens, labels, imgs):
        
        visual_proj = self.forward_visual(imgs)

        _bsz, seqlen = tokens.shape

        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[:seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers:
            h = layer(h, 0, freqs_cis, mask, visual_proj + adapter[adapter_index])
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h)
        output = output[:, :-1, :]
        labels = labels[:, 1:]

        if labels.sum() == 0:
            c_loss = output.mean() * 0
        else:
            assert self.llama.vocab_size == 32000
            c_loss = self.criterion(output.reshape(-1, self.llama.vocab_size), labels.flatten())

        return c_loss, c_loss

    #@torch.inference_mode()
    @torch.no_grad()
    def forward_inference(self, visual_proj, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]
        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)


        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0

        for layer in self.llama.layers:
            h = layer(h, start_pos, freqs_cis, mask, visual_proj + adapter[adapter_index].repeat(_bsz, 1, 1))
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float()

    #@torch.inference_mode()
    @torch.no_grad()
    def generate(
        self, imgs, prompts,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
    ):
        bsz = len(imgs)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        assert len(imgs) == len(prompts)
        
        with torch.cuda.amp.autocast():
            visual_query = self.forward_visual(imgs)

        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).cuda().long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            with torch.cuda.amp.autocast():
                logits = self.forward_inference(visual_query, tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded


_MODELS = {
    "BIAS-7B": "https://github.com/ZrrSkywalker/LLaMA-Adapter/releases/download/v.2.0.0/7fa55208379faf2dd862565284101b0e4a2a72114d6490a95e432cf9d9b6c813_BIAS-7B.pth",
    # "LORA16-7B": "",
    # "PARTIAL-7B": ""
}

def available_models():
    return list(_MODELS.keys())

def load(name, llama_dir, device="cuda" if torch.cuda.is_available() else "cpu", download_root='ckpts', max_seq_len=512,
        phase="finetune"):
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root)
    elif os.path.isfile(name):
        model_path = name
    else:
        return RuntimeError(f"Model {name} not found; available models = {available_models()}")

    ckpt = torch.load(model_path, map_location='cpu')
    
    # BIAS-7B or https://xxx/sha256_BIAS-7B.pth -> 7B
    llama_type = name.split('.')[0].split('-')[-1]
    llama_ckpt_dir = os.path.join(llama_dir, llama_type)
    llama_tokenzier_path = os.path.join(llama_dir, 'tokenizer.model')

    # load llama_adapter weights and model_cfg
    print(f'Loading LLaMA-Adapter from {model_path}')
    ckpt = torch.load(model_path, map_location='cpu')

    model = LLaMA_adapter(
        llama_ckpt_dir, llama_tokenzier_path,
        max_seq_len=max_seq_len, max_batch_size=1,
        clip_model='ViT-L/14@336px',
        v_embed_dim=1024, v_depth=16,
        v_num_heads=16, v_mlp_ratio=4.0,
        query_len=577, query_layer=32,
        phase=phase)

    load_result = model.load_state_dict(ckpt['model'], strict=False)

    # assert len(load_result.unexpected_keys) == 0, f"Unexpected keys: {load_result.unexpected_keys}"
    return model.to(device), model.clip_transform