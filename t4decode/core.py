"""Pure-Python data, scoring, configuration selection and persistence."""
from __future__ import annotations
import hashlib
import json
import math
import random
import statistics
import unicodedata
from pathlib import Path

PROFILES = {
    "smoke": dict(train=24, validation=9, test=9, perf=3, repeats=1, new_tokens=32, epochs=1),
    "standard": dict(train=600, validation=45, test=90, perf=12, repeats=3, new_tokens=128, epochs=2),
}
CANDIDATES = {
    "fp16": dict(bits=16, draft="none", lookahead=0),
    "int8": dict(bits=8, draft="none", lookahead=0),
    "nf4": dict(bits=4, draft="none", lookahead=0),
    "fp16_raw_k4": dict(bits=16, draft="raw", lookahead=4),
    "fp16_raw_dynamic": dict(bits=16, draft="raw", lookahead="dynamic"),
    "fp16_trained_k4": dict(bits=16, draft="trained", lookahead=4),
    "fp16_trained_dynamic": dict(bits=16, draft="trained", lookahead="dynamic"),
    "nf4_trained_dynamic": dict(bits=4, draft="trained", lookahead="dynamic"),
}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def source_digest():
    return digest({path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(Path(__file__).parent.glob("*.py"))})

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    # An interrupted last line is ignored; callers rewrite before resuming.
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise
    return rows

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()

def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).strip().casefold().split())

def score_answer(case, answer):
    if case.get("gold") is None:
        return None
    if case["task"] == "json":
        text = answer.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1])
        try:
            return float(json.loads(text) == case["gold"])
        except (ValueError, TypeError):
            return 0.0
    return float(normalize(answer) == normalize(str(case["gold"])))

def make_cases(split, count, seed=20261013, training=False):
    """Synthetic smoke/engineering corpus. Not a general Vietnamese benchmark."""
    salts = {"train": 101, "validation": 202, "test": 303}
    rng = random.Random(seed + salts[split])
    cases = []
    for i in range(count):
        tag = f"{split}-{i:05d}"
        records = [dict(code=f"{tag}-R{j}", room=rng.randint(101, 999), count=rng.randint(1, 50))
                   for j in range(rng.randint(4, 12))]
        chosen = rng.choice(records)
        source = "\n".join(f"Mã {r['code']}: phòng {r['room']}, số lượng {r['count']}." for r in records)
        kind = ("lookup", "sum", "json", "summary")[i % 4] if training else ("lookup", "sum", "json")[i % 3]
        if kind == "lookup":
            question = f"Mã {chosen['code']} ở phòng nào? Chỉ trả về số phòng, không giải thích."
            gold = str(chosen["room"])
        elif kind == "sum":
            question = "Tổng số lượng của tất cả mã là bao nhiêu? Chỉ trả về một số nguyên."
            gold = str(sum(r["count"] for r in records))
        elif kind == "json":
            question = f"Trích xuất thông tin mã {chosen['code']}. Chỉ trả JSON có hai khóa room và count, giá trị số nguyên."
            gold = dict(room=chosen["room"], count=chosen["count"])
        else:
            question = "Viết một đoạn tổng hợp dữ liệu này bằng tiếng Việt; nhắc đúng mã và số liệu, không thêm thông tin."
            gold = None
        cases.append(dict(id=tag, task=kind, prompt=f"Dữ liệu giả lập:\n{source}\n\n{question}", gold=gold))
    return cases

def make_perf_cases(split, count):
    """Fixed-output workloads, separate from natural-EOS quality evaluation."""
    cases = []
    for i in range(count):
        n = (4, 16, 40)[i % 3]
        source = "\n".join(
            f"Bản ghi {split}-{i}-{j}: một nhóm sinh viên đo thời gian xử lý, kiểm tra kết quả và ghi lỗi."
            for j in range(n))
        instruction = (
            "Từ dữ liệu, viết một hướng dẫn chi tiết để tổ chức thí nghiệm lập trình có thể tái lập.",
            "Tóm tắt nội dung và đề xuất quy trình kiểm tra chất lượng dữ liệu bằng tiếng Việt.",
            "Viết phần thảo luận có ví dụ về việc đo tốc độ và bộ nhớ của một chương trình."
        )[(i // 3) % 3]
        cases.append(dict(id=f"perf-{split}-{i:03d}", task="performance", prompt=source+"\n\n"+instruction, gold=None))
    return cases

def training_sequence(prompt_ids, answer_ids, max_length):
    if not answer_ids or len(prompt_ids) + len(answer_ids) > max_length:
        return None
    return dict(input_ids=list(prompt_ids)+list(answer_ids), labels=[-100]*len(prompt_ids)+list(answer_ids))

def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    index = (len(values)-1)*p
    lo, hi = math.floor(index), math.ceil(index)
    return values[lo] + (values[hi]-values[lo])*(index-lo)

def summarize(rows):
    result = {}
    for name in sorted({r["candidate"] for r in rows}):
        group = [r for r in rows if r["candidate"] == name]
        good = [r for r in group if r.get("status") == "ok"]
        perf = [r for r in good if r.get("kind") == "performance"]
        quality = [r["score"] for r in good if r.get("score") is not None]
        result[name] = dict(
            successful=len(good), errors=len(group)-len(good),
            quality_accuracy=statistics.mean(quality) if quality else None,
            quality_n=len(quality),
            latency_median_s=statistics.median([r["total_s"] for r in perf]) if perf else None,
            latency_p95_s=percentile([r["total_s"] for r in perf], .95),
            throughput_tok_s=(sum(r["output_tokens"] for r in perf)/sum(r["total_s"] for r in perf)) if perf else None,
            peak_vram_gib=max((r["peak_vram_gib"] for r in good), default=None),
            peak_ram_gib=max((r["peak_ram_gib"] for r in good), default=None),
            correctness=all(r.get("correctness", True) for r in group),
        )
    return result

def select_configuration(rows, budget_gib=12.0, allowed_quality_drop=.03):
    """Validation-only selection. No test access; incomplete candidates excluded."""
    summary = summarize(rows)
    ref = summary.get("fp16")
    if not ref or ref["quality_accuracy"] is None or ref["errors"]:
        raise ValueError("Baseline validation chưa hoàn tất hoặc không có điểm chất lượng.")
    ref_perf = {r["id"] for r in rows if r["candidate"] == "fp16" and r.get("kind") == "performance" and r.get("status") == "ok"}
    ref_quality = {r["id"] for r in rows if r["candidate"] == "fp16" and r.get("kind") == "quality" and r.get("status") == "ok"}
    eligible, reasons = [], {}
    for name, metrics in summary.items():
        group = [r for r in rows if r["candidate"] == name and r.get("status") == "ok"]
        perf_ids = {r["id"] for r in group if r.get("kind") == "performance"}
        quality_ids = {r["id"] for r in group if r.get("kind") == "quality"}
        why = []
        if metrics["errors"] or perf_ids != ref_perf or quality_ids != ref_quality:
            why.append("incomplete_or_failed")
        if not metrics["correctness"]:
            why.append("greedy_mismatch")
        if metrics["peak_vram_gib"] is None or metrics["peak_vram_gib"] > budget_gib:
            why.append("memory_budget")
        if metrics["quality_accuracy"] is None or metrics["quality_accuracy"] < ref["quality_accuracy"] - allowed_quality_drop:
            why.append("quality_gate")
        if metrics["latency_median_s"] is None:
            why.append("no_latency")
        reasons[name] = why
        if not why:
            eligible.append(name)
    if not eligible:
        raise ValueError(f"Không có cấu hình đạt yêu cầu: {reasons}")
    # Per-prompt paired geometric latency, so long prompts do not dominate selection.
    baseline = {r["id"]: r["total_s"] for r in rows if r["candidate"] == "fp16" and r.get("kind") == "performance" and r.get("status") == "ok"}
    ratios = {}
    for name in eligible:
        paired = [r["total_s"]/baseline[r["id"]] for r in rows if r["candidate"] == name and r.get("kind") == "performance" and r.get("status") == "ok"]
        ratios[name] = math.exp(statistics.mean(map(math.log, paired)))
    winner = min(eligible, key=lambda name: ratios[name])
    return dict(candidate=winner, config=CANDIDATES[winner], summary=summary, exclusions=reasons,
                paired_latency_ratios=ratios, budget_gib=budget_gib, allowed_quality_drop=allowed_quality_drop)
