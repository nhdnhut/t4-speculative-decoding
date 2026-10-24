"""Integration checks with random tiny Qwen2 weights; no model download/GPU.

These validate generation plumbing and telemetry, NOT Qwen3B/T4 performance.
"""
import copy
import unittest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from t4decode.engine import Engine

class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))

def make_engine(same=True):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        attention_dropout=0.0)
    engine = Engine.__new__(Engine)
    engine.model = Qwen2ForCausalLM(config).eval()
    engine.draft = copy.deepcopy(engine.model) if same else Qwen2ForCausalLM(config).eval()
    engine.tokenizer = TinyTokenizer()
    engine.encode = lambda prompt: torch.tensor([[1, 3, 9, 15]], dtype=torch.long)
    engine.cuda = False
    engine.max_context = 128
    engine.config = dict(lookahead=4)
    return engine

class EngineTests(unittest.TestCase):
    def test_exact_greedy_with_perfect_draft_and_trace(self):
        e = make_engine()
        direct = e.run("x", 16, fixed_length=True, assisted=False)
        assisted = e.run("x", 16, fixed_length=True, trace=True)
        self.assertEqual(direct["output_ids"], assisted["output_ids"])
        self.assertEqual(assisted["output_tokens"], 16)
        self.assertGreater(assisted["trace"]["proposed"], 0)
        self.assertEqual(assisted["trace"]["acceptance_rate"], 1.0)
        self.assertLess(assisted["target_forward_calls"], direct["target_forward_calls"])
        self.assertNotIn("_get_candidate_generator", e.model.__dict__)
        self.assertLess(assisted["ttft_s"], assisted["total_s"])

    def test_rejected_draft_keeps_greedy_output(self):
        e = make_engine(same=False)
        a = e.run("x", 12, fixed_length=True, assisted=False)
        b = e.run("x", 12, fixed_length=True, trace=True)
        self.assertEqual(a["output_ids"], b["output_ids"])
        self.assertGreaterEqual(b["trace"]["acceptance_rate"], 0.0)
        self.assertLessEqual(b["trace"]["acceptance_rate"], 1.0)

    def test_eos_is_counted_as_accepted_proposal(self):
        e = make_engine()
        next_id = e.run("x", 1, fixed_length=True, assisted=False)["output_ids"][0]
        e.model.generation_config.eos_token_id = next_id
        e.draft.generation_config.eos_token_id = next_id
        result = e.run("x", 16, trace=True)
        self.assertEqual(result["output_tokens"], 1)
        self.assertEqual(result["trace"]["acceptance_rate"], 1.0)

    def test_dynamic_and_context_guard(self):
        e = make_engine()
        e.config["lookahead"] = "dynamic"
        a = e.run("x", 9, fixed_length=True, assisted=False)
        b = e.run("x", 9, fixed_length=True)
        self.assertEqual(a["output_ids"], b["output_ids"])
        with self.assertRaises(ValueError):
            e.run("x", 129)

if __name__ == "__main__":
    unittest.main()
