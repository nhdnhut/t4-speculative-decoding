"""Rebuild all self-contained notebooks from the shared source (stdlib only)."""
import base64
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
payload = io.BytesIO()
with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
    for folder in ("t4decode", "tests"):
        for path in sorted((ROOT/folder).glob("*.py")):
            archive.writestr(str(path.relative_to(ROOT)), path.read_bytes())
    for name in ("requirements-colab.txt", "README.md"):
        archive.writestr(name, (ROOT/name).read_bytes())
raw = payload.getvalue()
encoded = base64.b64encode(raw).decode()
sha = hashlib.sha256(raw).hexdigest()

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip()+"\n"}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": textwrap.dedent(text).strip()+"\n"}

bootstrap = code(f'''
import base64, hashlib, io, os, subprocess, sys, zipfile
from pathlib import Path
SOURCE = Path("/content/t4decode_source")
blob = base64.b64decode("{encoded}")
assert hashlib.sha256(blob).hexdigest() == "{sha}", "Gói mã nguồn bị thay đổi"
SOURCE.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(io.BytesIO(blob)) as z:
    for member in z.namelist():
        assert (SOURCE/member).resolve().is_relative_to(SOURCE.resolve())
    z.extractall(SOURCE)
requirements = [s.strip() for s in (SOURCE/"requirements-colab.txt").read_text().splitlines()
                if s.strip() and not s.lstrip().startswith("#")]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + requirements)
sys.path.insert(0, str(SOURCE))
os.environ["PYTHONPATH"] = str(SOURCE) + os.pathsep + os.environ.get("PYTHONPATH", "")
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
print("Đã cài mã nguồn dùng chung:", SOURCE)
''')

setup = code('''
from google.colab import drive
drive.mount("/content/drive")
PROFILE = "standard"  # "smoke" cho lần kiểm tra ngắn; dùng project khác khi đổi profile
PROJECT = Path("/content/drive/MyDrive") / f"T4Decode_{PROFILE}"
PROJECT.mkdir(parents=True, exist_ok=True)

from t4decode.engine import environment
from t4decode.core import read_json
import json
print(json.dumps(environment(), ensure_ascii=False, indent=2))

def run_step(command, *extra):
    subprocess.check_call([sys.executable, "-m", "t4decode.cli", command,
                           "--project", str(PROJECT), *extra])

print("Thư mục dùng chung cả ba notebook:", PROJECT)
''')

qa = code('''
# Các phép kiểm tra này chỉ dùng Qwen2 ngẫu nhiên rất nhỏ trên CPU.
# Chúng kiểm tra thuật toán, không tạo số đo hiệu năng T4.
subprocess.check_call([sys.executable, "-m", "unittest", "discover", "-s", str(SOURCE/"tests"), "-v"])
''')

def display_latest(split):
    return code(f'''
import pandas as pd
from IPython.display import display, Markdown
run_dir = Path(read_json(PROJECT/"latest_{split}.json")["run"])
display(pd.read_csv(run_dir/"summary.csv"))
print("Dữ liệu thô:", run_dir)
''')

intro = '''
> Môi trường: **Colab → T4 GPU**. Giữ cùng thư mục `PROJECT` qua cả ba notebook.
> Notebook tự chứa mã nguồn; không cần upload thêm ZIP. Chạy từ trên xuống trên một runtime mới.
>
> **Trạng thái bàn giao:** đã kiểm tra mã và tích hợp bằng mô hình nhỏ trên CPU;
> chưa chạy huấn luyện Qwen2.5 hoặc benchmark T4 trong môi trường tạo notebook.
> Mọi bảng/biểu đồ bên dưới lấy từ kết quả thực khi bạn chạy, không có số liệu minh họa.
'''

nb1 = [
 md("# 01 · Baseline & Profiling trên T4\n\n"+intro),
 md("## 1. Cài môi trường và mã nguồn\n\nNếu đã import một phiên bản Transformers khác trước đó, hãy restart runtime trước cell này."),
 bootstrap, md("## 2. Gắn Drive và kiểm tra GPU\n\nT4 có 16 GB VRAM danh nghĩa. Chương trình ghi dung lượng khả dụng thực; dùng FP16, không ép BF16 hoặc FlashAttention-2."),
 setup, md("## 3. Kiểm tra đường chạy trước khi tải mô hình lớn"), qa,
 md("## 4. Khóa mô hình và dữ liệu\n\nModel SHA và hash train/validation/test được lưu một lần. `init` không ghi đè một project đã tồn tại."),
 code('run_step("init", "--profile", PROFILE)'),
 md("## 5. Đo baseline FP16\n\nKV cache bật, greedy, batch 1. Chất lượng dùng EOS tự nhiên. Phép đo hiệu năng sinh số token cố định và không dùng để chấm chất lượng. Baseline sẽ được đo lại ở notebook 03."),
 code('run_step("baseline")'), display_latest("validation"),
 md("## Chuyển sang notebook 02\n\nKết quả đã ghi vào Drive. Ngắt và xóa runtime này trước khi mở `02_Train_Optimize_T4.ipynb`. Đọc `README.md` trong mã nguồn để xem định nghĩa từng phép đo."),
]

nb2 = [
 md("# 02 · Train Draft & Optimize\n\n"+intro),
 md("## 1. Cài cùng môi trường"), bootstrap,
 md("## 2. Dùng đúng PROJECT của notebook 01"), setup,
 code('assert (PROJECT/"manifest.json").exists(), "Chạy notebook 01 trước."\nrun_step("init", "--profile", PROFILE)'),
 md('''
 ## 3. Tạo dữ liệu teacher có thể tiếp tục

 Target sinh đáp án train và validation. Các tác vụ có gold được chấm bằng đáp án độc lập;
 teacher sai bị loại khỏi dữ liệu train. Các đoạn tổng hợp tự do được gắn nhãn không có gold.
 Chạy lại cell sẽ bỏ qua mẫu đã lưu. Target được giải phóng trước khi train draft.
 '''),
 code('run_step("teacher")'),
 code('''
from t4decode.core import read_jsonl
for split in ("train", "validation"):
    rows = read_jsonl(PROJECT/"teacher"/f"{split}.jsonl")
    print(split, "đã tạo:", len(rows), "hợp lệ:", sum(r["keep"] for r in rows))
'''),
 md('''
 ## 4. LoRA draft 0.5B — lưu checkpoint và resume

 FP16, rank 16, gradient accumulation 8, response-only loss, LR 2e-4.
 Chạy lại cell sau khi runtime ngắt sẽ đọc checkpoint optimizer gần nhất.
 Loss nhỏ hơn không tự chứng minh tốc độ cao hơn: bước tiếp theo sẽ đo hệ thống thật.
 '''),
 code('run_step("train")'),
 code('''
import pandas as pd
import matplotlib.pyplot as plt
train_log = pd.DataFrame(read_jsonl(PROJECT/"training"/"training_log.jsonl"))
display(train_log.tail())
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(train_log["step"], train_log["train_loss"], label="Training")
if "validation_loss" in train_log:
    val = train_log.dropna(subset=["validation_loss"])
    ax.plot(val["step"], val["validation_loss"], "o-", label="Validation")
ax.set(xlabel="Optimizer step", ylabel="Cross-entropy loss", title="Loss của draft; không phải speedup")
ax.legend(); ax.grid(alpha=.2); plt.show()
'''),
 md('''
 ## 5. Đo tám cấu hình và chọn bằng validation

 FP16/INT8/NF4; draft có sẵn/đã train; lookahead 4/dynamic (tối đa 8).
 Mỗi cấu hình chạy trong tiến trình riêng, có warmup và kiểm tra greedy matching.
 Ngân sách ≤12 GiB và accuracy không giảm quá 0.03 tuyệt đối so với FP16 trên validation.
 Thử nghiệm được phép chọn FP16/plain nếu draft hoặc lượng tử hóa làm chậm.
 Chi phí tải model ghi riêng, không gộp vào tốc độ sinh token.
 '''),
 code('run_step("tune")'), display_latest("validation"),
 code('''
policy = read_json(PROJECT/"selected_policy.json")
print("Policy khóa từ validation:", policy["candidate"])
print("Cấu hình:", policy["config"])
print("Lý do loại các ứng viên:", json.dumps(policy["exclusions"], ensure_ascii=False, indent=2))
'''),
 md("## Chuyển sang notebook 03\n\nKhông sửa policy dựa trên test. Nếu đổi dữ liệu, precision hoặc recipe, hãy tạo một thí nghiệm có phiên bản riêng. Ngắt runtime này trước khi mở notebook 03."),
]

nb3 = [
 md("# 03 · Evaluation & Demo trên T4\n\n"+intro),
 md("## 1. Cài cùng môi trường"), bootstrap,
 md("## 2. Dùng đúng PROJECT và policy đã khóa"), setup,
 code('''
assert (PROJECT/"selected_policy.json").exists(), "Hoàn thành notebook 02 trước."
policy = read_json(PROJECT/"selected_policy.json")
print("Cấu hình khóa từ validation:", policy["candidate"])
'''),
 md('''
 ## 3. Chạy test mới và đo lại các đối chứng

 Mặc định chạy FP16, INT8, NF4, cấu hình được chọn và draft chưa train tương ứng nếu có.
 Bật FULL_ABLATION để đo đủ tám cấu hình. Không cập nhật policy từ kết quả test.
 Bộ test kỹ thuật giả lập không đại diện chất lượng tiếng Việt tổng quát; bổ sung benchmark thực
 trước khi khẳng định tính tổng quát trong báo cáo cuối kỳ.
 '''),
 code('''
FULL_ABLATION = False
run_step("evaluate", *(["--all-candidates"] if FULL_ABLATION else []))
'''), display_latest("test"),
 code('''
display(Markdown((run_dir/"REPORT.md").read_text()))
'''),
 md("## 4. Biểu đồ từ kết quả đo thực\n\nThroughput dưới đây bao gồm prefill + decoding, không phải tốc độ decode-only."),
 code('''
import matplotlib.pyplot as plt
table = pd.read_csv(run_dir/"summary.csv").set_index("candidate")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
table["throughput_tok_s"].plot.bar(ax=axes[0], color="#167D8D")
axes[0].set(title="Throughput thực tế", ylabel="Output tokens/s (gồm prefill)", xlabel="")
table["peak_vram_gib"].plot.bar(ax=axes[1], color="#D88C34")
axes[1].axhline(policy["budget_gib"], color="red", linestyle="--", label="Ngân sách lựa chọn")
axes[1].set(title="Peak VRAM đo được", ylabel="GiB", xlabel=""); axes[1].legend()
plt.tight_layout(); plt.savefig(run_dir/"comparison.png", dpi=180, bbox_inches="tight"); plt.show()
'''),
 md('''
 ## 5. Sử dụng hệ thống ngay trong Colab

 Cell này nạp cấu hình đã chọn một lần. Giao diện không mở public tunnel.
 Đo tổng thời gian generate và TTFT; văn bản hiển thị khi generate hoàn tất.
 Đây là demo greedy có giới hạn 256 token đầu ra; không dùng thời gian UI để tính benchmark.
 Nếu policy vượt giới hạn khi prompt mới dài hơn workload, ứng dụng báo lỗi để bạn điều chỉnh.
 '''),
 code('''
from t4decode.engine import Engine
from IPython.display import clear_output
import ipywidgets as widgets
ui_engine = Engine(PROJECT, policy["candidate"])
ui_engine.warmup()
question = widgets.Textarea(value="Giải thích KV cache bằng một ví dụ ngắn.",
    layout=widgets.Layout(width="100%", height="100px"))
button = widgets.Button(description="Sinh câu trả lời", button_style="primary")
release = widgets.Button(description="Giải phóng GPU")
out = widgets.Output()

def answer(_):
    button.disabled = True
    with out:
        clear_output(wait=True)
        try:
            result = ui_engine.run(question.value, max_new_tokens=256)
            print(result["text"])
            print(f"\\nGenerate: {result['total_s']:.2f}s | TTFT: {result['ttft_s']:.2f}s | "
                  f"VRAM: {result['peak_vram_gib']:.2f} GiB | {result['output_tokens']} token")
        except Exception as error:
            print(type(error).__name__, str(error))
    button.disabled = False

def free_gpu(_):
    ui_engine.close()
    button.disabled = True
    release.disabled = True
    with out:
        print("Đã giải phóng GPU. Chạy lại cell để mở demo mới.")

button.on_click(answer); release.on_click(free_gpu)
display(question, widgets.HBox([button, release]), out)
'''),
 md("## 6. Xuất dữ liệu báo cáo\n\nChỉ tải kết quả/manifest/policy; checkpoint và model giữ trong PROJECT để tránh ZIP quá lớn."),
 code('''
from google.colab import files
import zipfile
export = Path("/content/T4Decode_measured_results.zip")
with zipfile.ZipFile(export, "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("manifest.json", "selected_policy.json"):
        z.write(PROJECT/name, name)
    for p in run_dir.rglob("*"):
        if p.is_file():
            z.write(p, str(Path("test_run")/p.relative_to(run_dir)))
files.download(str(export))
'''),
]

for filename, cells in (("01_Baseline_T4.ipynb", nb1), ("02_Train_Optimize_T4.ipynb", nb2), ("03_Evaluate_Demo_T4.ipynb", nb3)):
    for i, cell in enumerate(cells):
        cell["id"] = f"cell-{i:03d}"
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells,
        metadata=dict(accelerator="GPU", kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                      language_info=dict(name="python", version="3.12"), colab=dict(name=filename, gpuType="T4", provenance=[]),
                      t4decode=dict(version="0.1.0", embedded_source_sha256=sha, gpu_benchmarks_executed=False)))
    (ROOT/"notebooks"/filename).write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(filename, len(cells), "cells")
