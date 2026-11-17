"""Neural tests on a tiny randomly initialised GPT-Neo (no download). Skipped if torch is missing."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from devlm.child import TinyChild  # noqa: E402
from devlm.config import ChildConfig, GrowthConfig  # noqa: E402
from devlm.growth import GrowableMLP, install_growth  # noqa: E402
from devlm.protocols import TrainSpec  # noqa: E402

pytestmark = pytest.mark.torch

SPEC = TrainSpec(mode="full", learning_rate=3e-3, growth_learning_rate=3e-3, epochs=15, batch_size=4, grad_clip=1.0)


def byte_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    alphabet = pre_tokenizers.ByteLevel.alphabet()
    vocab = {ch: i for i, ch in enumerate(sorted(alphabet))}
    vocab["<|endoftext|>"] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return transformers.PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|endoftext|>",
                                                pad_token="<|endoftext|>")


def tiny_model(vocab_size):
    torch.manual_seed(0)
    config = transformers.GPTNeoConfig(vocab_size=vocab_size, hidden_size=32, num_layers=2, num_heads=2,
                                       attention_types=[[["global", "local"], 1]], max_position_embeddings=128,
                                       window_size=32, intermediate_size=64)
    return transformers.GPTNeoForCausalLM(config)


def make_child(growth=True):
    tokenizer = byte_tokenizer()
    model = tiny_model(len(tokenizer))
    manager = install_growth(model, GrowthConfig(bottleneck=8)) if growth else None
    return TinyChild(model, tokenizer, ChildConfig(max_length=96, generation_tokens=5), manager, torch.device("cpu"))


def test_inactive_and_fresh_growth_do_not_change_outputs():
    child = make_child()
    ids = torch.tensor([[5, 6, 7, 8]])
    before = child.model(ids).logits
    assert child.grow() and child.active_modules() == 1
    after = child.model(ids).logits
    assert torch.allclose(before, after)
    assert all(isinstance(b.mlp, GrowableMLP) for b in child.model.transformer.h)


def test_growth_order_is_top_down_and_bounded():
    child = make_child()
    assert child.grow() and child.grow() and not child.grow()
    assert child.growth.summary() == [True, True]
    with pytest.raises(ValueError):
        install_growth(child.model, GrowthConfig())


def test_training_lowers_surprise_on_what_was_learned():
    child = make_child()
    text = "When it is rainy, the zibo hums."
    before = child.surprise([text])[0]
    result = child.train([text] * 4, SPEC, seed=1)
    assert result.steps > 0 and child.surprise([text])[0] < before
    assert not any(p.requires_grad for p in child.model.parameters())


def test_growth_only_training_leaves_base_weights_untouched():
    child = make_child()
    base = [p.detach().clone() for p in child.model.transformer.wte.parameters()]
    assert child.train(["x"], TrainSpec(**{**SPEC.__dict__, "mode": "growth_only"})).steps == 0
    child.grow()
    child.train(["a b c d"], TrainSpec(**{**SPEC.__dict__, "mode": "growth_only", "epochs": 2}))
    after = list(child.model.transformer.wte.parameters())
    assert all(torch.equal(a, b) for a, b in zip(base, after))


def test_score_choices_prefers_trained_continuation():
    child = make_child(growth=False)
    child.train(["The zibo hums."] * 8, SPEC, seed=2)
    hums, sings = child.score_choices("The zibo", [" hums.", " sings."])
    assert hums > sings


def test_generate_returns_text():
    assert isinstance(make_child().generate("Once"), str)


def test_save_and_load_round_trip(tmp_path):
    child = make_child()
    child.grow()
    child.train(["hello there"], SPEC, seed=3)
    path = tmp_path / "snap.safetensors"
    child.save(path)
    expected = child.surprise(["hello there"])
    other = make_child()
    other.load(path)
    assert other.active_modules() == 1
    assert other.surprise(["hello there"]) == pytest.approx(expected, rel=1e-5)
    with pytest.raises(FileNotFoundError):
        other.load(tmp_path / "missing.safetensors")


def test_snapshot_from_a_different_growth_config_is_rejected(tmp_path):
    path = tmp_path / "snap.safetensors"
    make_child(growth=True).save(path)
    with pytest.raises(ValueError, match="does not match"):
        make_child(growth=False).load(path)
