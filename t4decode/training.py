"""Offline sequence distillation, response-only LoRA, resumable optimizer steps."""
from __future__ import annotations
import math
import random
import shutil
import time
import uuid
from pathlib import Path
from .core import read_json, read_jsonl, write_json, write_jsonl, append_jsonl, digest, score_answer, training_sequence

def create_teacher(project):
    from .engine import Engine
    root = Path(project)
    engine = Engine(root, "fp16")
    try:
        for split in ("train", "validation"):
            path = root/"teacher"/f"{split}.jsonl"
            existing = read_jsonl(path)
            write_jsonl(path, existing)
            done = {r["id"] for r in existing}
            cases = read_jsonl(root/"data"/f"{split}.jsonl")
            for index, case in enumerate(cases):
                if case["id"] in done:
                    continue
                result = engine.run(case["prompt"], max_new_tokens=128)
                prompt_ids = engine.encode(case["prompt"])[0].tolist()
                quality = score_answer(case, result["text"])
                # Structured tasks have independent gold. Wrong teacher responses
                # are recorded, but excluded from distillation rather than relabeled.
                row = dict(id=case["id"], prompt_ids=prompt_ids, answer_ids=result["output_ids"],
                           text=result["text"], teacher_score=quality,
                           keep=bool(result["output_ids"]) and (quality is None or quality == 1.0))
                append_jsonl(path, row)
                print(f"teacher {split}: {index+1}/{len(cases)} keep={row['keep']}", flush=True)
    finally:
        engine.close()

def train_draft(project, resume=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    from .engine import environment, clean_cuda, PeakMonitor
    environment()
    root = Path(project)
    manifest = read_json(root/"manifest.json")
    folder = root/"training"
    folder.mkdir(parents=True, exist_ok=True)
    recipe = dict(seed=manifest["seed"], epochs=manifest["profile_settings"]["epochs"],
                  rank=16, alpha=32, lr=2e-4, grad_accumulation=8, max_length=1024,
                  checkpoint_every=10, weight_decay=.01)
    splits = {}
    for split in ("train", "validation"):
        rows = read_jsonl(root/"teacher"/f"{split}.jsonl")
        expected = len(read_jsonl(root/"data"/f"{split}.jsonl"))
        if len(rows) != expected:
            raise ValueError(f"Teacher {split} chưa hoàn tất: {len(rows)}/{expected}.")
        sequences = [training_sequence(r["prompt_ids"], r["answer_ids"], recipe["max_length"])
                     for r in rows if r["keep"]]
        splits[split] = [s for s in sequences if s is not None]
        print(f"{split}: dùng {len(splits[split])}/{len(rows)} mẫu (lọc teacher sai/quá dài).", flush=True)
    if len(splits["train"]) < 4 or len(splits["validation"]) < 1:
        raise ValueError("Không đủ mẫu teacher hợp lệ; xem teacher/*.jsonl.")
    signature = digest(dict(recipe=recipe, data=splits, model=manifest["draft"]))
    latest_file = folder/"latest.json"
    state = None
    if resume and latest_file.exists():
        checkpoint = folder/read_json(latest_file)["checkpoint"]
        state = torch.load(checkpoint/"state.pt", map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Dữ liệu/recipe đã đổi: tạo project mới để không resume sai thí nghiệm.")
    draft = manifest["draft"]
    tokenizer = AutoTokenizer.from_pretrained(draft["id"], revision=draft["revision"])
    target_tok = AutoTokenizer.from_pretrained(manifest["target"]["id"], revision=manifest["target"]["revision"])
    if tokenizer.get_vocab() != target_tok.get_vocab():
        raise ValueError("Token IDs teacher không tương thích tokenizer draft.")
    del target_tok
    torch.manual_seed(recipe["seed"])
    torch.cuda.manual_seed_all(recipe["seed"])
    base = AutoModelForCausalLM.from_pretrained(draft["id"], revision=draft["revision"],
                torch_dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa")
    if state:
        model = PeftModel.from_pretrained(base, str(checkpoint/"adapter"), is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=recipe["rank"], lora_alpha=recipe["alpha"],
                    lora_dropout=.05, target_modules="all-linear", task_type="CAUSAL_LM"))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    groups_per_epoch = math.ceil(len(splits["train"])/recipe["grad_accumulation"])
    total_steps = groups_per_epoch*recipe["epochs"]
    warmup = max(1, total_steps//10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:
        min((step+1)/warmup, max(0.0, (total_steps-step)/max(1, total_steps-warmup))))
    scaler = torch.amp.GradScaler("cuda")
    start_step, best_loss = 0, float("inf")
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_step, best_loss = state["step"], state["best_loss"]
        print(f"Resume optimizer step {start_step}/{total_steps}", flush=True)

    def batch(example):
        return {key: torch.tensor([value], dtype=torch.long, device="cuda") for key, value in example.items()} | {
            "attention_mask": torch.ones((1, len(example["input_ids"])), dtype=torch.long, device="cuda")}

    def validation_loss():
        model.eval()
        numerator, denominator = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for example in splits["validation"]:
                count = sum(x != -100 for x in example["labels"][1:])
                numerator += model(**batch(example)).loss.item()*count
                denominator += count
        model.train()
        return numerator/max(1, denominator)

    model.train()
    wall_start = time.perf_counter()
    log_file = folder/"training_log.jsonl"
    if state:
        write_jsonl(log_file, [r for r in read_jsonl(log_file) if r["step"] <= start_step])
    with PeakMonitor() as monitor:
        for epoch in range(recipe["epochs"]):
            order = list(range(len(splits["train"])))
            random.Random(recipe["seed"]+epoch).shuffle(order)
            for group_index in range(groups_per_epoch):
                step = epoch*groups_per_epoch+group_index+1
                if step <= start_step:
                    continue
                indices = order[group_index*recipe["grad_accumulation"]:(group_index+1)*recipe["grad_accumulation"]]
                optimizer.zero_grad(set_to_none=True)
                losses = []
                for index in indices:
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = model(**batch(splits["train"][index])).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Loss không hữu hạn; không lưu checkpoint lỗi.")
                    losses.append(loss.item())
                    scaler.scale(loss/len(indices)).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < old_scale
                if not skipped:
                    scheduler.step()
                row = dict(step=step, epoch=epoch+1, train_loss=sum(losses)/len(losses),
                           lr=scheduler.get_last_lr()[0], optimizer_step_skipped=skipped,
                           elapsed_this_session_s=time.perf_counter()-wall_start)
                if step % recipe["checkpoint_every"] == 0 or step == total_steps:
                    val_loss = validation_loss()
                    row["validation_loss"] = val_loss
                    if val_loss < best_loss:
                        best_loss = val_loss
                        model.save_pretrained(folder/"best_adapter", safe_serialization=True)
                        tokenizer.save_pretrained(folder/"best_adapter")
                    checkpoint = folder/f"checkpoint_{step:06d}"
                    checkpoint.mkdir(exist_ok=True)
                    model.save_pretrained(checkpoint/"adapter", safe_serialization=True)
                    torch.save(dict(signature=signature, step=step, best_loss=best_loss,
                            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), checkpoint/"state.pt")
                    write_json(latest_file, dict(checkpoint=checkpoint.name))
                    # Keep the newest two resumable checkpoints plus best adapter.
                    for old in sorted(folder.glob("checkpoint_*"))[:-2]:
                        shutil.rmtree(old)
                append_jsonl(log_file, row)
                print(row, flush=True)
    write_json(folder/"training_recipe.json", dict(recipe=recipe, signature=signature,
                  best_validation_loss=best_loss, **monitor.metrics()))
    del model, base, optimizer, scheduler, scaler
    clean_cuda()
    # Merge the best validation-loss adapter into the original FP16 draft.
    # Speed/acceptance are assessed separately by benchmark, not inferred from loss.
    base = AutoModelForCausalLM.from_pretrained(draft["id"], revision=draft["revision"], torch_dtype=torch.float16)
    best = PeftModel.from_pretrained(base, folder/"best_adapter")
    merged = best.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.save_pretrained(folder/"merged_draft", safe_serialization=True)
    tokenizer.save_pretrained(folder/"merged_draft")
    write_json(folder/"merged_draft"/"training_provenance.json", dict(signature=signature, base=draft, export_id=uuid.uuid4().hex))
    print("Đã lưu merged_draft. Chạy autotune để đo tốc độ thực tế.", flush=True)
