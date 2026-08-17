"""Module layer — core Module / ModuleList semantics, __str__, and edge cases."""
import pytest

from src.bten.tensor import Tensor, make_tensor
import src.front.functional as F
import src.front.module as nn


_counter = 0
def uid(prefix="t"):
    global _counter
    _counter += 1
    return f"{prefix}_{_counter}"


class TestModuleConstruction:
    """Module attribute registration and introspection."""

    @pytest.mark.unit
    def test_tensor_registration(self):
        m = nn.Module(uid("mod"))
        t = Tensor(uid("t"), shape=[3, 4], dtype='float32')
        m.my_tensor = t
        assert 'my_tensor' in m._tensors

    @pytest.mark.unit
    def test_op_handle_registration(self):
        m = nn.Module(uid("mod"))
        op = F.Gelu(uid("gelu"))
        m.my_op = op
        assert 'my_op' in m._op_hndls

    @pytest.mark.unit
    def test_submodule_registration(self):
        parent = nn.Module(uid("parent"))
        child = nn.Module(uid("child"))
        parent.child = child
        assert 'child' in parent._submodules

    @pytest.mark.unit
    def test_module_list_registration(self):
        parent = nn.Module(uid("parent"))
        children = nn.ModuleList([nn.Module(uid("c0")), nn.Module(uid("c1"))])
        parent.layers = children
        assert len(parent._submodules) == 2


class TestModuleList:

    @pytest.mark.unit
    def test_construction(self):
        mods = [nn.Module(uid("m")) for _ in range(3)]
        ml = nn.ModuleList(mods)
        assert len(ml) == 3

    @pytest.mark.unit
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            nn.ModuleList([])

    @pytest.mark.unit
    def test_none_module_raises(self):
        with pytest.raises(ValueError):
            nn.ModuleList([None])

    @pytest.mark.unit
    def test_indexing(self):
        mods = [nn.Module(uid("m")) for _ in range(3)]
        ml = nn.ModuleList(mods)
        assert ml[0] is mods[0]
        assert ml[-1] is mods[2]

    @pytest.mark.unit
    def test_slicing(self):
        mods = [nn.Module(uid("m")) for _ in range(4)]
        ml = nn.ModuleList(mods)
        subset = ml[1:3]
        assert len(subset) == 2

    @pytest.mark.unit
    def test_iteration(self):
        mods = [nn.Module(uid("m")) for _ in range(3)]
        ml = nn.ModuleList(mods)
        collected = list(ml)
        assert len(collected) == 3

    @pytest.mark.unit
    def test_immutable_setitem(self):
        mods = [nn.Module(uid("m")) for _ in range(2)]
        ml = nn.ModuleList(mods)
        with pytest.raises(RuntimeError):
            ml[0] = nn.Module(uid("m"))

    @pytest.mark.unit
    def test_immutable_delitem(self):
        mods = [nn.Module(uid("m")) for _ in range(2)]
        ml = nn.ModuleList(mods)
        with pytest.raises(RuntimeError):
            del ml[0]

    @pytest.mark.unit
    def test_immutable_append(self):
        mods = [nn.Module(uid("m")) for _ in range(2)]
        ml = nn.ModuleList(mods)
        with pytest.raises(RuntimeError):
            ml.append(nn.Module(uid("m")))

    @pytest.mark.unit
    def test_not_callable(self):
        mods = [nn.Module(uid("m")) for _ in range(2)]
        ml = nn.ModuleList(mods)
        with pytest.raises(RuntimeError):
            ml()

    @pytest.mark.unit
    def test_oob_indexing(self):
        mods = [nn.Module(uid("m")) for _ in range(2)]
        ml = nn.ModuleList(mods)
        with pytest.raises(IndexError):
            ml[5]


class TestModuleStr:
    """Module.__str__ should surface tensors/ops/submodules/params sections."""

    @pytest.mark.unit
    def test_str_with_tensors(self):
        m = nn.Module(uid("mod_str"))
        m.t1 = Tensor(uid("t1"), shape=[3, 4], dtype='float32')
        s = str(m)
        assert 'TENSORS' in s
        assert 't1' in s

    @pytest.mark.unit
    def test_str_with_ops(self):
        m = nn.Module(uid("mod_str2"))
        m.op1 = F.Gelu(uid("gelu_str"))
        s = str(m)
        assert 'OPS' in s

    @pytest.mark.unit
    def test_str_with_submodules(self):
        parent = nn.Module(uid("parent_str"))
        child = nn.Module(uid("child_str"))
        parent.child = child
        s = str(parent)
        assert 'SUBMODULES' in s

    @pytest.mark.unit
    def test_str_with_params(self):
        """LayerNorm has params on its lnorm op — __str__ should print PARAMS."""
        ln = nn.LayerNorm(uid("ln_str"), 64)
        s = str(ln)
        assert 'PARAMS' in s


class TestDropout:
    """Dropout is not in the op registry — both ctor signatures should raise."""

    @pytest.mark.unit
    def test_dropout_init_no_args(self):
        with pytest.raises(KeyError, match="Dropout"):
            nn.Dropout(uid("drop"))

    @pytest.mark.unit
    def test_dropout_init_with_prob(self):
        with pytest.raises(KeyError, match="Dropout"):
            nn.Dropout(uid("drop2"), prob=0.5)


class TestGetForwardGraph:

    @pytest.mark.unit
    def test_list_input(self):
        """get_forward_graph should accept list of tensors."""
        lin = nn.Linear(uid("lin_fg"), in_features=8, out_features=4)
        x = make_tensor(name=uid("x"), shape=[2, 8], dtype='float32')
        y = lin(x)
        G = lin.get_forward_graph([x])
        assert G.get_node_count() > 0

    @pytest.mark.unit
    def test_invalid_input_raises(self):
        """get_forward_graph with non-Tensor/non-list input should raise."""
        m = nn.Module(uid("mod_inv"))
        with pytest.raises((TypeError, AssertionError, UnboundLocalError)):
            m.get_forward_graph("not_a_tensor")


class TestModuleListExtraErrors:

    @pytest.mark.unit
    def test_invalid_type_index(self):
        mods = [nn.Module(uid("m")) for _ in range(2)]
        ml = nn.ModuleList(mods)
        with pytest.raises(TypeError):
            ml["invalid"]

    @pytest.mark.unit
    def test_extend_raises(self):
        mods = [nn.Module(uid("m")) for _ in range(2)]
        ml = nn.ModuleList(mods)
        with pytest.raises(RuntimeError):
            ml.extend([nn.Module(uid("m"))])

    @pytest.mark.unit
    def test_insert_raises(self):
        mods = [nn.Module(uid("m")) for _ in range(2)]
        ml = nn.ModuleList(mods)
        with pytest.raises(RuntimeError):
            ml.insert(0, nn.Module(uid("m")))
