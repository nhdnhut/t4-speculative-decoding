"""CLI used unchanged by all three notebooks. One worker process per candidate."""
from __future__ import annotations
import argparse
import csv
import datetime
import json
import os
import random
import subprocess
import sys
import uuid
from pathlib import Path
from .core import (PROFILES, CANDIDATES, digest, read_json, read_jsonl, write_json, write_jsonl,
                   append_jsonl, make_cases, make_perf_cases, score_answer, summarize, select_configuration, source_digest)

def initialize(root, profile):
    from huggingface_hub import HfApi
    from .engine import environment
    root.mkdir(parents=True, exist_ok=True)
    path = root/"manifest.json"
    if path.exists():
        manifest = read_json(path)
        if manifest["profile"] != profile:
            raise ValueError("Đổi profile bằng thư mục project mới; không ghi đè split/model lock.")
        print("Dùng manifest đã khóa:", path, flush=True)
        return
    settings = PROFILES[profile]
    api = HfApi()
    models = {}
    for key, model in (("target", "Qwen/Qwen2.5-3B-Instruct"), ("draft", "Qwen/Qwen2.5-0.5B-Instruct")):
        models[key] = dict(id=model, revision=api.model_info(model).sha)
    manifest = dict(version="0.1.0", source_sha=source_digest(), seed=20261013, profile=profile, profile_settings=settings,
                    max_context=4096, budget_gib=12.0, allowed_quality_drop=.03,
                    initial_environment=environment(), **models)
    hashes = {}
    for split in ("train", "validation", "test"):
        rows = make_cases(split, settings[split], training=split == "train")
        write_jsonl(root/"data"/f"{split}.jsonl", rows)
        hashes[split] = digest(rows)
        if split != "train":
            perf = make_perf_cases(split, settings["perf"])
            write_jsonl(root/"data"/f"{split}_perf.jsonl", perf)
            hashes[split+"_perf"] = digest(perf)
    manifest["data_hashes"] = hashes
    write_json(path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)

def check_data(root, manifest):
    if manifest.get("source_sha") != source_digest():
        raise ValueError("Mã nguồn khác manifest. Tạo project mới cho phiên bản mã mới.")
    for name, expected in manifest["data_hashes"].items():
        if digest(read_jsonl(root/"data"/f"{name}.jsonl")) != expected:
            raise ValueError(f"Split {name} bị sửa sau khi khóa manifest.")

def worker(root, candidate, split, output):
    from .engine import Engine, PeakMonitor, environment
    manifest = read_json(root/"manifest.json")
    check_data(root, manifest)
    settings = manifest["profile_settings"]
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, [])
    engine = None
    try:
        with PeakMonitor() as loading:
            engine = Engine(root, candidate)
        engine.warmup()
        cases = read_jsonl(root/"data"/f"{split}.jsonl")
        perf = read_jsonl(root/"data"/f"{split}_perf.jsonl")
        correct = True
        diagnostics = []
        if engine.draft is not None:
            # Same quantization, model weights, tokenizer, and generation rules.
            # Both natural EOS and fixed-length paths are tested before timing.
            for case, fixed in ((cases[0], False), (perf[0], True), (perf[-1], True)):
                direct = engine.run(case["prompt"], 32, fixed_length=fixed, assisted=False)
                assisted = engine.run(case["prompt"], 32, fixed_length=fixed, trace=True)
                match = direct["output_ids"] == assisted["output_ids"]
                correct = correct and match
                diagnostics.append(dict(id=case["id"], fixed_length=fixed, match=match,
                                        direct_ids=direct["output_ids"], assisted_ids=assisted["output_ids"],
                                        trace=assisted.get("trace")))
        write_json(output.with_suffix(".meta.json"), dict(environment=environment(), startup_s=engine.startup_s,
            startup_memory=loading.metrics(), diagnostics=diagnostics, model_lock=manifest["target"],
            candidate_config=CANDIDATES[candidate], data_hashes=manifest["data_hashes"]))
        if not correct:
            append_jsonl(output, dict(candidate=candidate, id="correctness", status="error", correctness=False,
                                     error="Greedy mismatch; xem diagnostics. Không công bố là lossless."))
            return
        # Shuffle prompt order deterministically. No prompt prefix cache is reused.
        jobs = [("quality", case, 0) for case in cases]
        repeats = settings["repeats"] if split == "test" else 1
        jobs += [("performance", case, repeat) for repeat in range(repeats) for case in perf]
        random.Random(manifest["seed"]).shuffle(jobs)
        for index, (kind, case, repeat) in enumerate(jobs):
            result = engine.run(case["prompt"], settings["new_tokens"] if kind == "performance" else 96,
                                fixed_length=kind == "performance")
            result["peak_vram_gib"] = max(result["peak_vram_gib"], loading.metrics()["peak_vram_gib"])
            result["peak_ram_gib"] = max(result["peak_ram_gib"], loading.metrics()["peak_ram_gib"])
            row = dict(candidate=candidate, id=f"{case['id']}/r{repeat}", case_id=case["id"], split=split,
                       kind=kind, repeat=repeat, task=case["task"], status="ok", correctness=correct,
                       score=score_answer(case, result["text"]) if kind == "quality" else None, **result)
            append_jsonl(output, row)
            if (index+1) % 10 == 0 or index+1 == len(jobs):
                print(f"{candidate} {split}: {index+1}/{len(jobs)}", flush=True)
    except Exception as exc:
        append_jsonl(output, dict(candidate=candidate, id="worker_failure", status="error", error=repr(exc)))
        raise
    finally:
        if engine is not None:
            engine.close()

def benchmark(root, split, names):
    from .engine import environment
    manifest = read_json(root/"manifest.json")
    check_data(root, manifest)
    if names == ["all"]:
        names = list(CANDIDATES)
    unknown = set(names)-set(CANDIDATES)
    if unknown:
        raise ValueError(f"Unknown candidates: {unknown}")
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:6]
    run = root/"runs"/f"{split}_{run_id}"
    run.mkdir(parents=True)
    order = list(names)
    random.Random(manifest["seed"]).shuffle(order)
    write_json(run/"run.json", dict(environment=environment(), split=split, order=order,
               manifest_digest=digest(manifest), started_at=run_id))
    all_rows = []
    for name in order:
        output = run/f"{name}.jsonl"
        result = subprocess.run([sys.executable, "-m", "t4decode.cli", "worker", "--project", str(root),
                "--candidate", name, "--split", split, "--output", str(output)],
                env=os.environ | {"PYTHONUNBUFFERED": "1"})
        rows = read_jsonl(output)
        if result.returncode and not any(r.get("status") == "error" for r in rows):
            rows.append(dict(candidate=name, id="process_failure", status="error", error=f"exit {result.returncode}"))
        all_rows.extend(rows)
    write_jsonl(run/"results.jsonl", all_rows)
    summary = summarize(all_rows)
    write_json(run/"summary.json", summary)
    if summary:
        with (run/"summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["candidate"]+list(next(iter(summary.values()))))
            writer.writeheader()
            for name, values in summary.items():
                writer.writerow(dict(candidate=name, **values))
    write_json(root/f"latest_{split}.json", dict(run=str(run)))
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return run, all_rows

def tune(root):
    manifest = read_json(root/"manifest.json")
    if not (root/"training"/"merged_draft"/"config.json").exists():
        raise FileNotFoundError("Hoàn thành teacher/train trước autotune để so sánh draft trước và sau train.")
    run, rows = benchmark(root, "validation", ["all"])
    policy = select_configuration(rows, manifest["budget_gib"], manifest["allowed_quality_drop"])
    policy.update(validation_run=str(run), manifest_digest=digest(manifest),
                  trained_provenance=read_json(root/"training"/"merged_draft"/"training_provenance.json"),
                  selected_on="validation_only", selection_unit="one_configuration_per_session")
    write_json(root/"selected_policy.json", policy)
    print("Cấu hình được chọn:", policy["candidate"], flush=True)

def evaluate(root, all_candidates=False):
    manifest = read_json(root/"manifest.json")
    policy = read_json(root/"selected_policy.json")
    if policy["manifest_digest"] != digest(manifest):
        raise ValueError("Manifest khác thời điểm autotune.")
    if policy["trained_provenance"] != read_json(root/"training"/"merged_draft"/"training_provenance.json"):
        raise ValueError("Draft đã được tạo lại sau autotune. Đo lại validation trước khi test.")
    # Always rerun FP16 + both ordinary quantized baselines, plus the selected
    # candidate and the trained/untrained counterpart (when it exists).
    names = ["fp16", "int8", "nf4", policy["candidate"]]
    chosen = policy["candidate"]
    if "trained" in chosen and chosen.replace("trained", "raw") in CANDIDATES:
        names.append(chosen.replace("trained", "raw"))
    names = list(CANDIDATES) if all_candidates else list(dict.fromkeys(names))
    run, rows = benchmark(root, "test", names)
    # Test results NEVER overwrite selected_policy.json.
    summary = summarize(rows)
    selected = summary.get(chosen, {})
    ref = summary.get("fp16", {})
    report = ["# Kết quả test T4Decode", "", f"Cấu hình khóa từ validation: `{chosen}`.",
              f"Run: `{run.name}`. Không tái chọn cấu hình bằng test.", "",
              "Tốc độ dưới đây gồm prefill + sinh fixed-length, không phải decode-only.",
              "Startup/download được ghi riêng trong *.meta.json.", "",
              "| Cấu hình | Accuracy | Tok/s | Median (s) | p95 (s) | VRAM peak GiB | Lỗi |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}"
    for name, s in summary.items():
        report.append(f"| {name} | {fmt(s['quality_accuracy'])} | {fmt(s['throughput_tok_s'])} | {fmt(s['latency_median_s'])} | {fmt(s['latency_p95_s'])} | {fmt(s['peak_vram_gib'])} | {s['errors']} |")
    report += ["", "## Giới hạn", "", "Corpus mặc định là dữ liệu giả lập có đáp án và các prompt hiệu năng tiếng Việt. "
        "Đây là kiểm thử kỹ thuật, không đại diện toàn bộ năng lực tiếng Việt. "
        "Thêm benchmark bên ngoài vào một project/dataset version mới để mở rộng kết luận.",
        "", "NVML lấy mẫu 20 ms có thể bỏ lỡ đỉnh ngắn. Bộ nhớ dành cho cả thiết bị có thể gồm tiến trình khác. "
        "Ngân sách 12 GiB là tiêu chí chọn cấu hình, không giả lập một GPU vật lý 12 GiB.",
        "", "Trace acceptance chạy riêng, không dùng thời gian có instrumentation để tính speedup.",
        "", "Loss của draft không chứng minh tốc độ. Chỉ đối chiếu các hàng cùng precision, cùng lookahead "
        "mới cô lập được tác động huấn luyện."]
    if selected.get("latency_median_s") and ref.get("latency_median_s"):
        report += ["", f"Tỷ số median baseline/selected: {ref['latency_median_s']/selected['latency_median_s']:.3f}x. "
                   "Đây là tỷ số median, không phải trung bình speedup từng prompt."]
    (run/"REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print("Báo cáo:", run/"REPORT.md", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "baseline", "teacher", "train", "tune", "evaluate", "worker"])
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, default="standard")
    parser.add_argument("--candidate", choices=CANDIDATES, default="fp16")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--all-candidates", action="store_true")
    args = parser.parse_args()
    root = args.project.resolve()
    if args.command == "init":
        initialize(root, args.profile)
    elif args.command == "baseline":
        benchmark(root, "validation", ["fp16"])
    elif args.command == "teacher":
        from .training import create_teacher
        create_teacher(root)
    elif args.command == "train":
        from .training import train_draft
        train_draft(root)
    elif args.command == "tune":
        tune(root)
    elif args.command == "evaluate":
        evaluate(root, args.all_candidates)
    elif args.command == "worker":
        if args.output is None:
            parser.error("worker needs --output")
        worker(root, args.candidate, args.split, args.output)

if __name__ == "__main__":
    main()
