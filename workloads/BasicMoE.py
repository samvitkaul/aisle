

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.sym import sym_ceil_mul
from src.front import make_front_tensor
import src.front.functional as F
import src.front.module as nn
import src.front.dynamic as D

from typing import Optional

class DenseExpert(nn.Module):
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.dE  : int = kwargs.get('dE')
        self.dFF : int = kwargs.get('dFF', 4*self.dE)

        self.ff1  = nn.Linear(self.name + '.ff1', self.dE, self.dFF)
        self.ff2  = nn.Linear(self.name + '.ff2', self.dFF, self.dE)
        self.act  = F.Gelu(self.name + '.gelu')

    def forward(self, x):
        return self.ff2( self.act ( self.ff1(x) ) )

class SwiGluExpert(nn.Module):
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.dE  : int = kwargs.get('dE')
        self.dFF : int = kwargs.get('dFF', 4*self.dE)

        self.gate  = nn.Linear(self.name + '.gate',  self.dE,  self.dFF)
        self.value = nn.Linear(self.name + '.value', self.dE,  self.dFF)
        self.down  = nn.Linear(self.name + '.down',  self.dFF, self.dE)
        self.act   = F.Sigmoid(self.name + '.sigmoid')

    def forward(self, x):
        g = self.gate(x)
        v = self.value(x)
        g = g * self.act(g)
        y = g * v
        y = self.down(y)
        return y

class DenseMoE(nn.Module):
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.dE       : int   = kwargs.get('dE')
        self.nE       : int   = kwargs.get('nE', 1) #num_experts
        self.aE       : int   = kwargs.get('aE', 1) #active experts
        self.sE       : int   = kwargs.get('sE', 0) #shared experts
        self.cFactor  : float = kwargs.get('capacity_factor', 1.0)
        self.exp_type : str   = kwargs.get('expert_type', 'swiglu')


        if self.aE > self.nE:
            raise ValueError(f'active experts({self.aE}) > total_experts({self.nE})')

        if self.sE < 0:
            raise ValueError(f'shared experts({self.sE}) < 0')

        if self.cFactor <= 0.0:
            raise ValueError(f'capacity_factor ({self.cFactor}) <= 0.0')

        if self.nE == 1:
            if self.aE != 1:
                raise ValueError(f'DenseMoE: nE == 1 requires aE == 1, got aE=({self.aE})')
            if self.sE != 0:
                raise ValueError(f'DenseMoE: nE == 1 requires sE == 0, got sE=({self.sE})')

        self.exp_cls = SwiGluExpert if self.exp_type.upper() == 'SWIGLU' else DenseExpert
        self._mlp_mode = (self.nE == 1)

        if not self._mlp_mode:
            self.router = nn.Linear(self.name + '.router', self.dE, self.nE)
        self.experts = nn.ModuleList([
            self.exp_cls(self.name + f'.expert_{i}', **kwargs) for i in range(self.nE)
            ])
        if self.sE > 0:
            self.shared_experts = nn.ModuleList([
                self.exp_cls(self.name + f'.shared_expert_{i}', **kwargs) for i in range(self.sE)
                ])

    def forward(self, x):
        batch, seqlen, hidden_dim = x.shape

        if hidden_dim != self.dE:
            raise ValueError(f'input hidden_dim({hidden_dim}) != Module.dE({self.dE})')

        if self._mlp_mode:
            return self.experts[0](x)

        num_tokens    = batch * seqlen
        tokens        = x.reshape(num_tokens, self.dE) #flatten [num_tokens, dE]
        router_logits = self.router(tokens) #[num_tokens, nE]

        """
           NOTE: A faithful sparse topk MoE dispatch would do
              mask == (active_indices == expert_id)
              positions = mask.nonzero(...)
              selected = tokens[positions]; ... ; output[positions] += ...
           but the static graph frontend currently has no Equal/NonZero ops,
           no data-dependent Gather, and not FrontTensor __setitem__
           We don't have a path today to lower a data dependent routing loop
           into the trace. therefore we trace a mathemactically equivalent
           *dense* combine: every expert sees every token and per-expert contributions
           are weighted by a full softmax over router_logits. This preserves the op-graph
           shape that the cost/perf model care about (router -> softmax -> nE expert subgraphs
           -> weighted sum -> optional shared experts) while only using ops the registry
           exposes (Softmax, Slice, Mul, Add)
        """
        full_weights = router_logits.softmax(dim=-1) #[num_tokens, nE]
        
        #expert capacity (static -- no dynamic drop possible in trace)
        total_assignments   = num_tokens * self.aE
        expert_capacity     = sym_ceil_mul(total_assignments, self.cFactor, divisor=self.nE)
        dropped_assignments = 0

        #Dense Dispatch + Combine
        output = None
        for expert_id, expert in enumerate(self.experts):
            expert_output = expert(tokens)                                     #[num_tokens, dE]
            w_expert      = full_weights[:, expert_id:expert_id+1]             #[num_tokens,  1]
            contrib       = w_expert * expert_output                           #[num_tokens, dE]
            output        = contrib if output is None else (output + contrib)

        if self.sE > 0:
            shared_output = None
            for expert in self.shared_experts:
                so = expert(tokens)
                shared_output = so if shared_output is None else (shared_output + so)
            if self.sE > 1:
                #average shared-expert contributions via a const scalar Mul
                # (FrontTensor.__truediv__ requires a Tensor RHS, not a Python int)
                inv_sE = make_front_tensor(
                        name= self.name + '.shared_inv_count',
                        shape=[],
                        data= 1.0/self.sE,
                        dtype=tokens.dtype.name,
                        is_const=True
                        )
                shared_output = shared_output * inv_sE
            output = output + shared_output

        output = output.reshape(batch, seqlen, self.dE)

        auxinfo = dict(
                expert_capacity=expert_capacity,
                dropped_assignments=dropped_assignments,
                )
        print(auxinfo)

        return output

    def inputs(self, bs=1, seqlen=8, dE=32):
        tokens = make_front_tensor(name='tokens', shape=[bs, seqlen, dE], dtype='bfloat16')
        return (tokens,)

class SparseMoE(nn.Module):
    """
       Single Device sparse MoE with static capacity factor dispatch

       Dataflow (one realistic dispatch->compute->combine trace)
          Router -> Softmax -> TopK
               -> Gather (dispatch tokens to experts)    [N, dE] -> [nE, capacity, dE]
               -> Split + Squeeze (per expert slice)     nE x [capacity, dE]
               -> Expert(.) per expert                   nE x [capacity, dE]
               -> Unsqueeze + Concat (stack)             [nE, capacity, dE]
               -> Mul with combile weights               [nE, capacity,  1] x [nE, capacity, dE]
               -> ScatterND combine into zero buffer     [N, dE]
          (+ optional shared experts)
          -> Reshape back to [bs, seqlen, dE]

       Drop in API compatible with DenseMoE

       dispatch_indices and combine_weights are declared const tensors with data=None so 
       graph2onnx auto fills random placeholders at export time. The trace's shape and byte
       movement are faithful - but numeric output is not comparable to any reference
       PyTorch MoE forward pass
    """
    def __init__(self, name, **kwargs):
        super().__init__(name)
        self.dE       : int   = kwargs.get('dE')
        self.nE       : int   = kwargs.get('nE', 1) #num_experts
        self.aE       : int   = kwargs.get('aE', 1) #active experts
        self.sE       : int   = kwargs.get('sE', 0) #shared experts
        self.cFactor  : float = kwargs.get('capacity_factor', 1.0)
        self.exp_type : str   = kwargs.get('expert_type', 'swiglu')


        if self.aE > self.nE:
            raise ValueError(f'active experts({self.aE}) > total_experts({self.nE})')

        if self.sE < 0:
            raise ValueError(f'shared experts({self.sE}) < 0')

        if self.cFactor <= 0.0:
            raise ValueError(f'capacity_factor ({self.cFactor}) <= 0.0')

        if self.nE == 1:
            if self.aE != 1:
                raise ValueError(f'DenseMoE: nE == 1 requires aE == 1, got aE=({self.aE})')
            if self.sE != 0:
                raise ValueError(f'DenseMoE: nE == 1 requires sE == 0, got sE=({self.sE})')

        self.exp_cls = SwiGluExpert if self.exp_type.upper() == 'SWIGLU' else DenseExpert
        self._mlp_mode = (self.nE == 1)

        if not self._mlp_mode:
            self.router = nn.Linear(self.name + '.router', self.dE, self.nE)
            self.expert_split = F.Split(self.name + '.expert_split', axis=0, num_outputs=self.nE)
            self.combine_mul  = F.Mul(self.name + '.combine_mul')
            self.combine_scatter = F.ScatterND(self.name + '.combine_scatter')

        self.experts = nn.ModuleList([
            self.exp_cls(self.name + f'.expert_{i}', **kwargs) for i in range(self.nE)
            ])

        if self.sE > 0:
            self.shared_experts = nn.ModuleList([
                self.exp_cls(self.name + f'.shared_expert_{i}', **kwargs) for i in range(self.sE)
                ])

    def forward(self, x):
        batch, seqlen, hidden_dim = x.shape

        if hidden_dim != self.dE:
            raise ValueError(f'input hidden_dim({hidden_dim}) != Module.dE({self.dE})')

        if self._mlp_mode:
            return self.experts[0](x)

        num_tokens    = batch * seqlen
        tokens        = x.reshape(num_tokens, self.dE)   #[N, dE]
        router_logits = self.router(tokens)              #[N, nE]
        router_probs  = router_logits.softmax(dim=-1)    #[N, nE]

        def _reg(t):
            self._tensors[t.name] = t
            return t

        _topk_values, _topk_indices = router_probs.topk(k=self.aE) #[N, aE] each

        total_assignments   = num_tokens * self.aE
        expert_capacity     = sym_ceil_mul(total_assignments, self.cFactor, divisor=self.nE)
        dropped_assignments = 0

        dispatch_indices = _reg(make_front_tensor(
            name=self.name + '.dispatch_indices',
            shape=[self.nE, expert_capacity],
            dtype='int64',
            is_const=True))

        combine_weights = _reg(make_front_tensor(
            name=self.name + '.combine_weights',
            shape=[self.nE, expert_capacity, 1],
            dtype=tokens.dtype.name,
            is_const=True))

        scatter_indices = _reg(make_front_tensor(
            name=self.name + '.scatter_indices',
            shape=[self.nE, expert_capacity, 1],
            dtype='int64',
            is_const=True))
        zeros_buf = _reg(make_front_tensor(
            name=self.name + '.combine_zeros',
            shape=[num_tokens, self.dE],
            dtype=tokens.dtype.name,
            is_const=True))

        #Dispatch: [N, dE] gather [nE, capacity] -> [nE, capacity, dE]
        dispatched = tokens[dispatch_indices]

        #Per expert compute: Split(axis=0) -> nE x [1, capacity, dE]
        per_expert_slices = self.expert_split(dispatched)

        expert_outputs = []
        for expert_id, expert in enumerate(self.experts):
            #Squeeze [1,capacity,dE] -> [capacity,dE]
            slice_e = per_expert_slices[expert_id].squeeze(dim=0)
            out_e = expert(slice_e)
            expert_outputs.append(out_e)

        #Stack: nE x [capacity,dE] -> [nE,capacity,dE]
        expert_out = D.stack(expert_outputs, dim=0)

        #Weight + combine: ScatterND into zero buffer -> [N,dE]
        weighted = self.combine_mul(combine_weights, expert_out)
        combined = self.combine_scatter(zeros_buf, scatter_indices, weighted)

        #Optional shared experts
        if self.sE > 0:
            shared_output = None
            for expert in self.shared_experts:
                so = expert(tokens)
                shared_output = so if shared_output is None else (shared_output + so)
            if self.sE > 1:
                inv_sE = make_front_tensor(
                        name= self.name + '.shared_inv_count',
                        shape=[],
                        data= 1.0/self.sE,
                        dtype=tokens.dtype.name,
                        is_const=True
                        )
                shared_output = shared_output * inv_sE
            combined = combined + shared_output

        output = combined.reshape(batch, seqlen, self.dE)

        auxinfo = dict(
                expert_capacity=expert_capacity,
                dropped_assignments=dropped_assignments,
                dispatch_op='gather',
                combine_op='scatternd',
                )
        print(auxinfo)

        return output

    def inputs(self, bs=1, seqlen=8, dE=32):
        tokens = make_front_tensor(name='tokens', shape=[bs, seqlen, dE], dtype='bfloat16')
        return (tokens,)


if __name__ == '__main__':
    from src.graph import graph2onnx

    bs, seqlen = 2, 8
    cfg = dict(dE=32, dFF=128, expert_type='swiglu', nE=8, aE=2, sE=1, capacity_factor=1.25)

    M   = DenseMoE('DenseMoE', **cfg)
    Xs  = M.inputs(bs=bs, seqlen=seqlen, dE=cfg['dE'])
    Y   = M(*Xs)
    for x in Xs: print('IN', x)
    print('OUT', Y)
    G = M.get_forward_graph(*Xs)
    print('Dumping DenseMoE.onnx')
    graph2onnx(G, 'DenseMoE.onnx')

    print('-'*30)

    SM  = SparseMoE('SparseMoE', **cfg)
    SXs = SM.inputs(bs=bs, seqlen=seqlen, dE=cfg['dE'])
    SY  = SM(*SXs)
    for x in SXs: print('IN', x)
    print('OUT', SY)
    SG = SM.get_forward_graph(*SXs)
    print('Dumping SparseMoE.onnx')
    graph2onnx(SG, 'SparseMoE.onnx')

