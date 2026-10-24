import tempfile
import unittest
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import Qwen2Config, Qwen2ForCausalLM
from t4decode.core import training_sequence

class LoRATests(unittest.TestCase):
    def test_response_only_backward_save_reload_and_merge(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, pad_token_id=0)
        base = Qwen2ForCausalLM(cfg)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base.save_pretrained(root/"base")
            model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", task_type="CAUSAL_LM"))
            seq = training_sequence([1,3,8], [9,11,2], 64)
            data = {k: torch.tensor([v]) for k,v in seq.items()}
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
            loss = model(**data).loss
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad))
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
            opt.step()
            model.eval()
            before = model(input_ids=data["input_ids"]).logits.detach()
            model.save_pretrained(root/"adapter")
            loaded = PeftModel.from_pretrained(Qwen2ForCausalLM.from_pretrained(root/"base"), root/"adapter").eval()
            merged = loaded.merge_and_unload(safe_merge=True).eval()
            after = merged(input_ids=data["input_ids"]).logits.detach()
            torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-4)

if __name__ == "__main__":
    unittest.main()
