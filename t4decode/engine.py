"""HF inference backend. No implicit CPU offload; FP16 compute on NVIDIA T4."""
from __future__ import annotations
import contextlib
import gc
import importlib.metadata
import os
import platform
import threading
import time
from pathlib import Path
from .core import CANDIDATES, read_json

def environment(require_t4=True):
    import torch
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "tokenizers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    info = dict(python=platform.python_version(), platform=platform.platform(), packages=versions,
                cuda_runtime=torch.version.cuda, cuda_available=torch.cuda.is_available())
    if not torch.cuda.is_available():
        if require_t4:
            raise RuntimeError("Chọn Colab Runtime > Change runtime type > T4 GPU trước khi chạy.")
        return info
    prop = torch.cuda.get_device_properties(0)
    info.update(gpu=prop.name, compute_capability=[prop.major, prop.minor], total_vram_gib=prop.total_memory/2**30)
    if require_t4 and "T4" not in prop.name:
        raise RuntimeError(f"Notebook chuẩn hóa cho T4; GPU đang được cấp là {prop.name}. Đổi sang T4.")
    return info

def clean_cuda():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

class PeakMonitor:
    """Sample process RSS / whole-device NVML usage; allocator peak is exact for PyTorch.

    NVML peaks sampled every 20 ms may miss a transient spike. Other GPU processes
    contribute to device_used. It is a measurement gate, not a hardware memory cap.
    """
    def __init__(self, cuda=True):
        import psutil
        try:
            self.process = psutil.Process()
        except psutil.Error:
            self.process = None
        self.cuda = cuda
        self.stop = threading.Event()
        self.rss = 0
        self.device_used = 0
        self.nvml = None
        if cuda:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml = pynvml
            except Exception:
                self.nvml = None

    def sample(self):
        if self.process is not None:
            self.rss = max(self.rss, self.process.memory_info().rss)
        else:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            self.rss = max(self.rss, rss*(1 if platform.system() == "Darwin" else 1024))
        if self.nvml:
            try:
                self.device_used = max(self.device_used, self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used)
            except Exception:
                pass

    def __enter__(self):
        import torch
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.sample()
        def loop():
            while not self.stop.wait(.02):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        import torch
        if self.cuda:
            torch.cuda.synchronize()
        self.stop.set()
        self.thread.join()
        self.sample()
        self.allocated = torch.cuda.max_memory_allocated() if self.cuda else 0
        self.reserved = torch.cuda.max_memory_reserved() if self.cuda else 0

    def metrics(self):
        return dict(peak_ram_gib=self.rss/2**30,
                    ram_source="sampled_RSS" if self.process is not None else "process_lifetime_highwater",
                    peak_allocated_gib=self.allocated/2**30,
                    peak_reserved_gib=self.reserved/2**30,
                    peak_device_used_gib=self.device_used/2**30 if self.nvml else None,
                    peak_vram_gib=max(self.device_used, self.reserved)/2**30,
                    memory_source="NVML_device_and_torch" if self.nvml else "torch_allocator_only")

class FirstDelivery:
    """Token delivery timing, not text chunk timing. Prompt put() is excluded."""
    def __init__(self):
        self.prompt_seen = False
        self.first_time = None
        self.first_tokens = 0

    def put(self, value):
        tokens = value.detach().cpu()
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        if self.first_time is None and tokens.numel():
            self.first_time = time.perf_counter()
            self.first_tokens = tokens.numel()

    def end(self):
        pass

@contextlib.contextmanager
def candidate_trace(target):
    """Diagnostic-only adapter for the pinned HF candidate-generator API.

    Separate from timed runs. Counts accepted proposals, including all-accepted
    EOS/max-length blocks. Restores methods even if generation fails.
    """
    original = target._get_candidate_generator
    had_override = "_get_candidate_generator" in target.__dict__
    stats = dict(proposed=0, accepted=0, rounds=0)
    def factory(*args, **kwargs):
        generator = original(*args, **kwargs)
        get = generator.get_candidates
        update = generator.update_candidate_strategy
        last = {}
        def get_candidates(input_ids):
            ids, logits = get(input_ids)
            last.update(start=input_ids.shape[1], proposed=ids.shape[1]-input_ids.shape[1], ids=ids)
            return ids, logits
        def update_strategy(input_ids, scores, num_matches):
            proposed = last["proposed"]
            committed = input_ids.shape[1]-last["start"]
            accepted = min(proposed, max(0, int(num_matches)))
            if proposed and committed == proposed and bool((input_ids[0, -1] == last["ids"][0, -1]).item()):
                accepted = proposed
            stats["proposed"] += proposed
            stats["accepted"] += accepted
            stats["rounds"] += 1
            return update(input_ids, scores, num_matches)
        generator.get_candidates = get_candidates
        generator.update_candidate_strategy = update_strategy
        return generator
    target._get_candidate_generator = factory
    try:
        yield stats
    finally:
        if had_override:
            target._get_candidate_generator = original
        else:
            del target._get_candidate_generator

class Engine:
    def __init__(self, project, candidate="fp16"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        environment()
        self.project = Path(project)
        self.manifest = read_json(self.project/"manifest.json")
        self.name = candidate
        self.config = CANDIDATES[candidate]
        self.cuda = True
        started = time.perf_counter()
        target = self.manifest["target"]
        self.tokenizer = AutoTokenizer.from_pretrained(target["id"], revision=target["revision"])
        self.tokenizer.padding_side = "left"
        kwargs = dict(torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa")
        if self.config["bits"] == 8:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.config["bits"] == 4:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(target["id"], revision=target["revision"], **kwargs).eval()
        self.draft = None
        if self.config["draft"] != "none":
            draft = self.manifest["draft"]
            if self.config["draft"] == "trained":
                path = self.project/"training"/"merged_draft"
                if not (path/"config.json").exists():
                    raise FileNotFoundError("Chạy notebook 02 để tạo merged_draft trước.")
                source, revision = str(path), None
            else:
                source, revision = draft["id"], draft["revision"]
            draft_tok = AutoTokenizer.from_pretrained(source, revision=revision)
            if draft_tok.get_vocab() != self.tokenizer.get_vocab():
                raise ValueError("Target/draft không cùng ánh xạ token; không dùng speculative trực tiếp.")
            self.draft = AutoModelForCausalLM.from_pretrained(
                source, revision=revision, torch_dtype=torch.float16,
                device_map={"": 0}, attn_implementation="sdpa").eval()
        self.startup_s = time.perf_counter()-started
        self.max_context = self.manifest["max_context"]
        self.last_ids = []

    def encode(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": "Bạn là trợ lý. Làm đúng yêu cầu và chỉ dùng dữ liệu được cung cấp khi câu hỏi yêu cầu."},
             {"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
            return_tensors="pt")

    def run(self, prompt, max_new_tokens=128, fixed_length=False, assisted=True, trace=False):
        import torch
        from transformers import GenerationConfig
        request_start = time.perf_counter()
        ids = self.encode(prompt).to(self.model.device)
        prompt_len = ids.shape[1]
        if prompt_len + max_new_tokens > self.max_context:
            raise ValueError(f"Context {prompt_len}+{max_new_tokens}>{self.max_context}; không tự cắt prompt.")
        streamer = FirstDelivery()
        forward_calls = [0]
        hook = self.model.register_forward_pre_hook(lambda module, args: forward_calls.__setitem__(0, forward_calls[0]+1))
        gc = GenerationConfig.from_model_config(self.model.config)
        gc.do_sample = False
        gc.num_beams = 1
        gc.max_new_tokens = max_new_tokens
        # In HF 4.57.1 min_new_tokens creates a MinLengthLogitsProcessor that
        # assisted generation rejects. Suppress EOS equally for all fixed-length
        # candidates instead; natural-EOS quality runs keep standard stopping.
        gc.min_new_tokens = None
        gc.min_length = 0
        gc.use_cache = True
        gc.eos_token_id = self.model.generation_config.eos_token_id
        if fixed_length and gc.eos_token_id is not None:
            gc.suppress_tokens = gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]
        gc.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        kwargs = {}
        use_draft = self.draft is not None and assisted
        if use_draft:
            dynamic = self.config["lookahead"] == "dynamic"
            self.draft.generation_config.num_assistant_tokens = 8 if dynamic else self.config["lookahead"]
            self.draft.generation_config.num_assistant_tokens_schedule = "constant"
            self.draft.generation_config.assistant_confidence_threshold = .4 if dynamic else 0.0
            kwargs["assistant_model"] = self.draft
        tracker = candidate_trace(self.model) if trace and use_draft else contextlib.nullcontext(None)
        try:
            with PeakMonitor(self.cuda) as peak, torch.inference_mode(), tracker as stats:
                if self.cuda:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                output = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                             generation_config=gc, use_model_defaults=False, streamer=streamer, **kwargs)
                if self.cuda:
                    torch.cuda.synchronize()
                end = time.perf_counter()
        finally:
            hook.remove()
        token_ids = output[0, prompt_len:].detach().cpu().tolist()
        self.last_ids = token_ids
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        result = dict(text=text, output_ids=token_ids, input_tokens=prompt_len, output_tokens=len(token_ids),
                      total_s=end-start, request_total_s=time.perf_counter()-request_start,
                      ttft_s=streamer.first_time-start if streamer.first_time is not None else None,
                      first_delivery_tokens=streamer.first_tokens, target_forward_calls=forward_calls[0],
                      tokens_per_target_forward=len(token_ids)/max(1, forward_calls[0]),
                      throughput_tok_s=len(token_ids)/max(end-start, 1e-9), **peak.metrics())
        if stats is not None:
            result["trace"] = dict(**stats, acceptance_rate=stats["accepted"]/stats["proposed"] if stats["proposed"] else None)
            result["timing_is_diagnostic"] = True
        return result

    def warmup(self):
        self.run("Chỉ viết từ Xin chào.", max_new_tokens=8, fixed_length=True)

    def close(self):
        self.model = None
        self.draft = None
        clean_cuda()
