
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src import make_tensor
import src.front.functional as F
import src.front.module as nn
import src.front.dynamic as D

import math

class ATTN(nn.Module):
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.dE           = kwargs.get('dE')
        self.nH           = kwargs.get('nH')
        self.idE          = kwargs.get('idE', 4*self.dE)
        self.dH           = self.dE // self.nH

        #tensors
        self.attn_sqrt_dH = make_tensor(name=self.name + '.sqrt_dH',
                                        shape=[],
                                        dtype='float32',
                                        data=math.sqrt(self.dH))

        #ops
        self.wqkv_proj  = nn.Linear(self.name +'.wqkv_proj',  self.dE, 3*self.dE)
        self.wqkv_split = F.Split  (self.name +'.wqkv_split', num_outputs=3, axis=2)
        self.w0_proj    = nn.Linear(self.name +'.w0_proj', self.dE, self.dE)


    def forward(self, x, past_kv=None, use_cache=False):
        batch, seqlen, hidden_dim = x.shape
        assert hidden_dim == self.dE, f"Input hidden_dim= {hidden_dim} != {self.name}.dE= {self.dE}"
        WQKV  = self.wqkv_proj(x)
        Q,K,V = self.wqkv_split(WQKV)
        Q     = Q.reshape(batch, seqlen, self.nH, self.dH).transpose(1,2)
        K     = K.reshape(batch, seqlen, self.nH, self.dH).transpose(1,2)
        V     = V.reshape(batch, seqlen, self.nH, self.dH).transpose(1,2)

        past_seqlen = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            K = D.cat([past_k, K], dim=2)
            V = D.cat([past_v, V], dim=2)
            past_seqlen = past_k.size(2)

        QK    = (Q @ K.transpose(2,3)) * self.attn_sqrt_dH

        #TODO: add trilu/mask dynamically
        #mask = F.triu(make_ones_tensor(seqlen, past_seqlen + seqlen), diagonal=past_seqlen + 1)
        #QK = QK.masked_fill(mask==1, float('-inf'))

        QK  = QK.softmax(dim=-1)
        QKV = (QK @ V).transpose(1,2).reshape(batch, seqlen, self.dE)

        present_kv = (K, v) if use_cache else None
        output     = self.w0_proj(QKV)
        return output, present_kv

class TransformerBlock(nn.Module):
    def __init__(self, name, **cfg):
        super().__init__(name)
        self.dE          = cfg.get('dE')
        self.nH          = cfg.get('nH')
        self.idE         = cfg.get('idE', 4*self.dE)

        #ops
        self.ln_attn_in  = nn.LayerNorm (self.name + '.ln_attn_in', self.dE)
        self.attn        = ATTN         (self.name + '.attn', **cfg)
        self.lnorm       = nn.LayerNorm (self.name + '.ln_mlp_in', self.dE)
        self.ff1         = nn.Linear    (self.name + '.ff1', self.dE, self.idE)
        self.ff2         = nn.Linear    (self.name + '.ff2', self.idE, self.dE)
        self.gelu        = F.Gelu       (self.name + '.gelu')

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, present_kv = self.attn(self.ln_attn_in(x), past_kv, use_cache)
        y = x + attn_out
        y = self.lnorm(y)
        y = self.ff1(y)
        y = self.gelu(y)
        y = self.ff2(y)
        return y, present_kv

class BasicLLM(nn.Module):
    def __init__(self, name, **cfg):
        super().__init__(name)
        self.vocab_sz     = cfg.get('vocab_sz')
        self.dE           = cfg.get('dE')
        self.nW           = cfg.get('nW')
        self.nH           = cfg.get('nH')
        self.nL           = cfg.get('nL')
        self.nL_proxy     = 1

        #ops
        self.wte     = nn.Embedding (self.name+'.wte', self.vocab_sz, self.dE)
        self.wpe     = nn.Embedding (self.name+'.wpe', self.nW, self.dE)
        self.tblocks = nn.ModuleList([
            TransformerBlock(f'{self.name}t{i}', **cfg) for i in range(self.nL_proxy)
            ])
        self.head = nn.Linear(self.name + '.head', self.dE, self.vocab_sz)


    def forward(self, input_tokens, past_kv=None, use_cache=False):
        batch_size, seqlen = input_tokens.shape

        positions = make_tensor(name='positions', shape=[batch_size, seqlen], dtype='int64')
        self._tensors[positions.name] = positions

        Y = self.wte(input_tokens)
        Z = self.wpe(positions)

        if past_kv is None:
            past_kv = [None] * self.nL_proxy

        present_kv = []
        for i,tblock in enumerate(self.tblocks):
            Y, pres_kv = tblock(Y, past_kv[i], use_cache)
            present_kv.append(pres_kv)

        logits = self.head(Y)

        #update repeat counts
        for tblock in self.tblocks:
            repeated_ops: dict[str, Any] = {}
            tblock.get_ops(repeated_ops)
            for op_name,op_obj in repeated_ops.items():
                op_obj.repeat_count = self.nL

        present_kv = present_kv if use_cache else None
        return logits, present_kv

    def inputs(self, bs=1):
        tokens = make_tensor(name='tokens', shape=[bs, 9], dtype='int64')
        return (tokens,)

if __name__ == '__main__':
    from src.graph import graph2onnx

    cfg = dict(nL=3, nH=3, dE=48, nW=32, vocab_sz=50257)
    gpt_nano = BasicLLM('gpt_nano', **cfg)
    itensors = gpt_nano.inputs()
    otensors = gpt_nano(*itensors)
    G        = gpt_nano.get_forward_graph(*itensors)
    graph2onnx(G, 'BasicLLM.onnx')
