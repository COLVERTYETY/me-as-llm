import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import inspect # for checking if we can use fused versions of torch.optim accoding to device type
from collections import defaultdict, Counter

class MLP(nn.Module):

    def __init__(self, n_embed, dropout=0.2,exp=2):
        super().__init__()
        self.c_fc    = nn.Linear(n_embed, exp * n_embed, bias=False)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(exp * n_embed, n_embed, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.2,exp=2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.w1 = nn.Linear( n_embd, exp * n_embd, bias=False)
        self.w2 = nn.Linear( exp * n_embd, n_embd, bias=False)
        self.w3 = nn.Linear( n_embd, exp*n_embd, bias=False)

    def forward(self, x) -> torch.Tensor:
        x = nn.functional.silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        return self.dropout(x)

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


class CausalSelfAttention(nn.Module):

    def __init__(self, block_size, n_embed, n_head, dropout=0.2, flash_attention=True):
        super().__init__()
        # assert n_embed % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embed, 3 * n_embed, bias=False)
        # output projection
        self.c_proj = nn.Linear(n_embed, n_embed, bias=True)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embed
        self.dropout = dropout
        self.flash_attention = flash_attention
        if not flash_attention:
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.flash_attention:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / torch.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):

    def __init__(self, block_size, n_embed, n_head, flash_attention=True):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(block_size, n_embed, n_head, dropout=0.2, flash_attention=flash_attention)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class MBlock(nn.Module):

    def __init__(self, block_size, n_embed, n_head, flash_attention=True):
        super().__init__()
        self.ln_1 = RMSNorm(n_embed)
        self.attn = CausalSelfAttention(block_size, n_embed, n_head, dropout=0.2, flash_attention=flash_attention)
        self.ln_2 = RMSNorm(n_embed)
        self.mlp = FeedForward(n_embed)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class llm(nn.Module):
    def __init__(self, vocab_size:int, block_size:int, n_embed:int, n_head:int, n_layers:int, flash_attention=True) -> None:
        super().__init__()
        self.block_size = block_size
        self.n_embed = n_embed
        self.n_head = n_head
        self.flash_attention = flash_attention

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embed), # word token embeddings
            wpe = nn.Embedding(block_size, n_embed), # positional embeddings
            drop = nn.Dropout(0.2),
            h = nn.ModuleList([Block(block_size, n_embed, n_head, flash_attention) for _ in range(n_layers)]),
            ln_f = nn.LayerNorm(n_embed),
        ))
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, inputs_idx, labels=None):
        dev = inputs_idx.device
        input_shape = inputs_idx.size()
        b, t = input_shape
        # assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        pos = torch.arange(t, dtype=torch.long, device=dev)

        # forward the GPT model
        tok_emb = self.transformer.wte(inputs_idx) # each index maps to a (learnable) vector
        pos_emb = self.transformer.wpe(pos) # each position maps to a (learnable) vector

        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if labels is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
        else:
            # infernec-time we only lm_head on the last token of each sequence
            logits = self.lm_head(x[:, [-1], :]) # [-1] preserves the "time" dim
            loss = None

        return logits, loss
    
    def crop_block_size(self, block_size):
            # model surgery to decrease the block size if necessary
            # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
            # but want to use a smaller block size for some smaller, simpler model
            assert block_size <= self.block_size
            self.block_size = block_size
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
            for block in self.transformer.h:
                if hasattr(block.attn, 'bias'):
                    block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, repetition_penalty=1.1,top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Repetition penalty is applied to decrease the likelihood of repeating tokens.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            #  count token repetitions and apply repetition penalty
            counts = Counter(idx.view(-1).tolist())
            # avg = sum(counts.values()) / len(counts)
            for k,v in counts.items():
                logits[:,k] /= repetition_penalty**(v)

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


class Mllm(nn.Module):
    def __init__(self, vocab_size:int, block_size:int, n_embed:int, n_head:int, n_layers:int, flash_attention=True) -> None:
        super().__init__()
        self.block_size = block_size
        self.n_embed = n_embed
        self.n_head = n_head
        self.flash_attention = flash_attention

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embed), # word token embeddings
            wpe = nn.Embedding(block_size, n_embed), # positional embeddings
            drop = nn.Dropout(0.2),
            h = nn.ModuleList([MBlock(block_size, n_embed, n_head, flash_attention) for _ in range(n_layers)]),
            ln_f = RMSNorm(n_embed),
        ))
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, inputs_idx, labels=None):
        dev = inputs_idx.device
        input_shape = inputs_idx.size()
        b, t = input_shape
        # assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        pos = torch.arange(t, dtype=torch.long, device=dev)

        # forward the GPT model
        tok_emb = self.transformer.wte(inputs_idx) # each index maps to a (learnable) vector
        pos_emb = self.transformer.wpe(pos) # each position maps to a (learnable) vector

        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if labels is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
        else:
            # infernec-time we only lm_head on the last token of each sequence
            logits = self.lm_head(x[:, [-1], :]) # [-1] preserves the "time" dim
            loss = None

        return logits, loss
    
    def crop_block_size(self, block_size):
            # model surgery to decrease the block size if necessary
            # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
            # but want to use a smaller block size for some smaller, simpler model
            assert block_size <= self.block_size
            self.block_size = block_size
            self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
            for block in self.transformer.h:
                if hasattr(block.attn, 'bias'):
                    block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, repetition_penalty=1.1,top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Repetition penalty is applied to decrease the likelihood of repeating tokens.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            #  count token repetitions and apply repetition penalty
            counts = Counter(idx.view(-1).tolist())
            # avg = sum(counts.values()) / len(counts)
            for k,v in counts.items():
                logits[:,k] /= repetition_penalty**(v)

            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx