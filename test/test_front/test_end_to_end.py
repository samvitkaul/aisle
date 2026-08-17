"""End-to-end front-end traces — composed nn.Module workloads."""
import math
from itertools import pairwise

import pytest

import src.front.functional as F
import src.front.module as nn
from src import make_tensor

_counter = 0
def uid(prefix="t"):
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


# ---------------------------------------------------------------------------
# Reusable mini-models
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    """A 2-layer MLP: Linear → Gelu → Linear → Sigmoid."""
    def __init__(self, name, d_in, d_hidden, d_out):
        super().__init__(name)
        self.fc1  = nn.Linear(name + '.fc1', d_in, d_hidden)
        self.act1 = F.Gelu(name + '.gelu')
        self.fc2  = nn.Linear(name + '.fc2', d_hidden, d_out)
        self.act2 = F.Sigmoid(name + '.sigmoid')

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x

    def inputs(self):
        x = make_tensor(name='x', shape=[4, 32], dtype='float32')
        return (x,)


class SimpleAttention(nn.Module):
    """Minimal single-head attention: Q·Kᵀ → softmax → ·V."""
    def __init__(self, name, d_model):
        super().__init__(name)
        self.d_model = d_model
        self.wq = nn.Linear(name + '.wq', d_model, d_model)
        self.wk = nn.Linear(name + '.wk', d_model, d_model)
        self.wv = nn.Linear(name + '.wv', d_model, d_model)
        self.wo = nn.Linear(name + '.wo', d_model, d_model)
        self.softmax = F.Softmax(name + '.softmax')
        self.scale = make_tensor(name=name + '.scale', shape=[], dtype='float32',
                                 data=1.0 / math.sqrt(d_model))

    def forward(self, x):
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)
        # Q @ K^T
        K_t = K.transpose(-2, -1)
        scores = Q @ K_t
        scores = scores * self.scale
        attn = self.softmax(scores)
        out = attn @ V
        return self.wo(out)

    def inputs(self):
        x = make_tensor(name='attn_input', shape=[1, 8, 64], dtype='float32')
        return (x,)


class ResBlock(nn.Module):
    """Linear → Gelu → Linear + residual skip."""
    def __init__(self, name, dim):
        super().__init__(name)
        self.fc1 = nn.Linear(name + '.fc1', dim, 4 * dim)
        self.act = F.Gelu(name + '.gelu')
        self.fc2 = nn.Linear(name + '.fc2', 4 * dim, dim)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act(h)
        h = self.fc2(h)
        return h + x  # residual

    def inputs(self):
        x = make_tensor(name='resblock_input', shape=[2, 10, 64], dtype='float32')
        return (x,)


class TransformerLayer(nn.Module):
    """LN → Attention → residual → LN → FFN → residual."""
    def __init__(self, name, d_model):
        super().__init__(name)
        self.ln1  = nn.LayerNorm(name + '.ln1', d_model)
        self.attn = SimpleAttention(name + '.attn', d_model)
        self.ln2  = nn.LayerNorm(name + '.ln2', d_model)
        self.ffn  = ResBlock(name + '.ffn', d_model)

    def forward(self, x):
        h = self.ln1(x)
        h = self.attn(h)
        x = x + h
        h = self.ln2(x)
        h = self.ffn(h)
        # Note: ResBlock already adds residual internally, but here we add
        # the LN output to the original.  For the testbench the exact
        # architecture doesn't matter — we're testing that the front-end
        # composes correctly.
        return x + h

    def inputs(self):
        x = make_tensor(name='tl_input', shape=[1, 8, 64], dtype='float32')
        return (x,)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEndToEnd:

    @pytest.mark.unit
    def test_simple_mlp(self):
        mlp = SimpleMLP(uid("mlp"), d_in=32, d_hidden=128, d_out=10)
        x_tup = mlp.inputs()
        y = mlp(*x_tup)
        assert y.shape == [4, 10]

    @pytest.mark.unit
    def test_mlp_graph(self):
        mlp = SimpleMLP(uid("mlp"), d_in=32, d_hidden=128, d_out=10)
        x_tup = mlp.inputs()
        y = mlp(*x_tup)
        G = mlp.get_forward_graph(*x_tup)
        assert G.get_node_count() > 0
        assert G.get_edge_count() > 0

    @pytest.mark.unit
    def test_simple_attention(self):
        attn = SimpleAttention(uid("attn"), d_model=64)
        x_tup = attn.inputs()
        y = attn(*x_tup)
        assert y.shape == [1, 8, 64]

    @pytest.mark.unit
    def test_attention_graph(self):
        attn = SimpleAttention(uid("attn"), d_model=64)
        x_tup = attn.inputs()
        y = attn(*x_tup)
        G = attn.get_forward_graph(*x_tup)
        assert G.get_node_count() > 0

    @pytest.mark.unit
    def test_residual_block(self):
        res = ResBlock(uid("res"), dim=64)
        x_tup = res.inputs()
        y = res(*x_tup)
        assert y.shape == [2, 10, 64]

    @pytest.mark.unit
    def test_residual_graph(self):
        res = ResBlock(uid("res"), dim=64)
        x_tup = res.inputs()
        y = res(*x_tup)
        G = res.get_forward_graph(*x_tup)
        assert G.get_node_count() > 0
        ordered = G.get_ordered_nodes()
        assert len(ordered) > 0

    @pytest.mark.unit
    def test_transformer_layer(self):
        tl = TransformerLayer(uid("tl"), d_model=64)
        x_tup = tl.inputs()
        y = tl(*x_tup)
        assert y.shape == [1, 8, 64]

    @pytest.mark.unit
    def test_transformer_layer_graph(self):
        tl = TransformerLayer(uid("tl"), d_model=64)
        x_tup = tl.inputs()
        y = tl(*x_tup)
        G = tl.get_forward_graph(*x_tup)
        assert G.get_node_count() > 0
        assert G.get_edge_count() > 0

    @pytest.mark.unit
    def test_stacked_linear_layers(self):
        """Dynamic loop over layers, similar to BasicMLP workload."""
        class StackedLinear(nn.Module):
            def __init__(self, name, dims):
                super().__init__(name)
                #for i, (m, n) in enumerate(zip(dims[:-1], dims[1:])):
                for i, (m, n) in enumerate(pairwise(dims)):
                    setattr(self, f'fc{i}', nn.Linear(f'{name}.fc{i}', m, n))
                    setattr(self, f'act{i}', F.Gelu(f'{name}.gelu{i}'))
                self.num_layers = len(dims) - 1

            def forward(self, x):
                for i in range(self.num_layers):
                    x = getattr(self, f'fc{i}')(x)
                    x = getattr(self, f'act{i}')(x)
                return x

            def inputs(self):
                x = make_tensor(name='stack_input', shape=[2, 5, 32],
                                dtype='float32')
                return (x,)

        model = StackedLinear(uid("stack"), dims=[32, 128, 256, 64, 10])
        x_tup = model.inputs()
        y = model(*x_tup)
        assert y.shape == [2, 5, 10]
        G = model.get_forward_graph(*x_tup)
        assert G.get_node_count() > 0

    @pytest.mark.unit
    def test_view_in_attention_pattern(self):
        """reshape + transpose pattern from multi-head attention."""
        class MHAReshape(nn.Module):
            def __init__(self, name, d_model, n_heads):
                super().__init__(name)
                self.d_model = d_model
                self.n_heads = n_heads
                self.d_head = d_model // n_heads
                self.proj = nn.Linear(name + '.proj', d_model, d_model)

            def forward(self, x):
                B, S, _ = x.shape
                h = self.proj(x)                       # [B, S, D]
                h = h.view(B, S, self.n_heads, self.d_head)  # [B, S, nH, dH]
                h = h.transpose(1, 2)                  # [B, nH, S, dH]
                return h

            def inputs(self):
                x = make_tensor(name='mha_input', shape=[1, 8, 64],
                                dtype='float32')
                return (x,)

        model = MHAReshape(uid("mha"), d_model=64, n_heads=4)
        x_tup = model.inputs()
        y = model(*x_tup)
        assert y.shape == [1, 4, 8, 16]
