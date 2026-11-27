"""Derive a time-budgeted recipe without modifying V4 or V5 Score Focus."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib

from tools.build_kaggle_v5_score_focus import build_notebook as score_notebook, code, unpack_bundle

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle_team34_v5_balanced.ipynb"


def build_notebook():
    notebook = score_notebook()
    source, node, bundle = unpack_bundle(notebook)
    bundle["src/balanced_budget.py"] = (ROOT / "src/balanced_budget.py").read_text(encoding="utf-8")
    raw = base64.b64encode(zlib.compress(json.dumps(bundle, ensure_ascii=False).encode("utf-8"), 9)).decode("ascii")
    lines = source.splitlines(keepends=True)
    assignment = "_BUNDLE_B64 = (\n" + "\n".join(repr(x) for x in textwrap.wrap(raw, 100)) + "\n)\n"
    source = "".join(lines[:node.lineno-1]) + assignment + "".join(lines[node.end_lineno:])
    notebook["cells"][2] = code(source)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        text = "".join(cell["source"])
        for old, new in (("work_kaggle_v5", "work_kaggle_v5_balanced"),
                         ("/kaggle/working/repo_v5", "/kaggle/working/repo_v5_balanced"),
                         ("/kaggle/tmp/v5_raw", "/kaggle/tmp/v5_balanced_raw"),
                         ("/kaggle/working/staged_v5", "/kaggle/working/staged_v5_balanced"),
                         ("/kaggle/working/golden_val_v5", "/kaggle/working/golden_val_v5_balanced"),
                         ("master_v5", "master_v5_balanced"), ("du_lieu_v5", "du_lieu_v5_balanced"),
                         ("/kaggle/working/outputs_v5", "/kaggle/working/outputs_v5_balanced"),
                         ("/kaggle/working/runs_v5", "/kaggle/working/runs_v5_balanced"),
                         ("/kaggle/working/runs_stage1_v5", "/kaggle/working/runs_stage1_v5_balanced"),
                         ("team34_v5_score", "team34_v5_balanced")):
            text = text.replace(old, new)
        cell.update(code(text))
    params = "".join(notebook["cells"][1]["source"])
    params = params.replace('import os\n', 'import os\nimport time\nRUN_STARTED = time.monotonic()\n', 1)
    for old, new in (("TRAIN_EPOCHS = 110", "TRAIN_EPOCHS = 65"),
                     ("STAGE1_EPOCHS = 35", "STAGE1_EPOCHS = 15"),
                     ("REFINE_EPOCHS = 25", "REFINE_EPOCHS = 8"),
                     ("TRAIN_BATCH = 8", "TRAIN_BATCH = 16"),
                     ("TRAIN_PATIENCE = 30", "TRAIN_PATIENCE = 15"),
                     ("MAPILLARY_MAX_IMAGES = 3200", "MAPILLARY_MAX_IMAGES = 2800"),
                     ("BDD_MAX_IMAGES = 400", "BDD_MAX_IMAGES = 600"),
                     ("KAGGLE_MAX_IMAGES_PER_SOURCE = 900", "KAGGLE_MAX_IMAGES_PER_SOURCE = 500"),
                     ("VNTS_MAX_IMAGES = 800", "VNTS_MAX_IMAGES = 400"),
                     ("MAX_EXTRA_CROPS = 800", "MAX_EXTRA_CROPS = 400"),
                     ("COPY_PASTE_COUNT = 300", "COPY_PASTE_COUNT = 200"),
                     ("WEATHER_COUNT = 150", "WEATHER_COUNT = 75")):
        if old not in params:
            raise ValueError(f"Recipe changed: expected {old}")
        params = params.replace(old, new)
    params += '''
# All sources still use V5 label-quality rules. COCO is omitted to avoid a full 25GB load plus completion pass.
AUTO_DOWNLOAD_KAGGLE_SLUGS = [s for s in AUTO_DOWNLOAD_KAGGLE_SLUGS if "coco-2017" not in s]
SKIP_STOCK_INPUT_SLUGS = SKIP_STOCK_INPUT_SLUGS + ["coco-2017-dataset"]
TOTAL_BUDGET_HOURS = 7.5
EXPORT_RESERVE_HOURS = .75
TEACHER_CAP_HOURS = .65
MAIN_CAP_HOURS = 3.0
REFINE_CAP_HOURS = .35
RUN_LOCAL_TRAIN_QC = False  # Keep full held-out official evaluation; defer diagnostic inference on 500 training images.
print("V5 Balanced: 65 main epochs + up to 8 refine; elapsed time includes data preparation.")
'''
    notebook["cells"][1] = code(params)
    completion = "".join(notebook["cells"][8]["source"])
    completion = completion.replace('from src.build_master import build_master_dataset',
        'from src.build_master import build_master_dataset\nfrom src.balanced_budget import phase_hours, StopAtDeadline')
    completion = completion.replace('        s1_model.train(data=',
        '''        teacher_hours = phase_hours(RUN_STARTED, TOTAL_BUDGET_HOURS, EXPORT_RESERVE_HOURS + 1.5, TEACHER_CAP_HOURS)
        if teacher_hours < .10:
            raise RuntimeError("Insufficient time to train teacher and student; use saved staged data in a new session")
        s1_model.add_callback("on_train_batch_end", StopAtDeadline(teacher_hours))
        s1_model.train(data=''')
    completion = completion.replace('lr0=.001, cos_lr=True, mask_ratio=2,', 'lr0=.001, cos_lr=True, mask_ratio=4, save_period=5,')
    notebook["cells"][8] = code(completion)
    train = "".join(notebook["cells"][11]["source"])
    train = train.replace('from src.score_focus_v5 import make_repeat_factor_dataset, choose_official_candidate, holdout_fingerprint',
        'from src.score_focus_v5 import make_repeat_factor_dataset, choose_official_candidate, holdout_fingerprint\nfrom src.balanced_budget import phase_hours, StopAtDeadline')
    train = train.replace('max_extra_fraction=.35', 'max_extra_fraction=.20')
    train = train.replace('mask_ratio=2, overlap_mask=True, max_det=100,', 'mask_ratio=4, overlap_mask=True, max_det=100, save_period=5,')
    train = train.replace('    model.train(data=', '''    main_hours = phase_hours(RUN_STARTED, TOTAL_BUDGET_HOURS, EXPORT_RESERVE_HOURS, MAIN_CAP_HOURS)
    if main_hours < .25:
        raise RuntimeError("Insufficient training time remaining after preparation; keep staged data and restart")
    model.add_callback("on_train_batch_end", StopAtDeadline(main_hours))
    model.train(data=''', 1)
    train = train.replace('mosaic=.6, close_mosaic=20,', 'mosaic=.6, close_mosaic=12,')
    train = train.replace('    if RUN_CLEAN_REFINEMENT:\n', '''    # Export a validated recovery submission BEFORE optional refinement.
    exported_candidates = {"main": export_and_verify(main_best, OUTPUTS_DIR / "candidates/main",
        notebook_path=REPO_DIR / "train_and_export.ipynb", smoke_images_dir=MASTER_DIR / "images/val")}
    shutil.copy2(OUTPUTS_DIR / "candidates/main/submission.zip", OUTPUTS_DIR / "recovery_main_submission.zip")
    refine_hours = phase_hours(RUN_STARTED, TOTAL_BUDGET_HOURS, EXPORT_RESERVE_HOURS, REFINE_CAP_HOURS)
    if RUN_CLEAN_REFINEMENT and refine_hours >= .10:
''')
    train = train.replace('        refine.train(data=', '''        refine.add_callback("on_train_batch_end", StopAtDeadline(refine_hours))
        refine_common = dict(common, mask_ratio=2, batch=min(8, TRAIN_BATCH))
        refine.train(data=''', 1)
    train = train.replace('name=RUN_NAME+"_refine", **common)', 'name=RUN_NAME+"_refine", **refine_common)')
    train = train.replace('        exported = export_and_verify(checkpoint, candidate_dir, notebook_path=REPO_DIR / "train_and_export.ipynb",\n            smoke_images_dir=MASTER_DIR / "images/val")',
        '''        exported = exported_candidates.get(name)
        if exported is None:
            exported = export_and_verify(checkpoint, candidate_dir, notebook_path=REPO_DIR / "train_and_export.ipynb",
                smoke_images_dir=MASTER_DIR / "images/val")''')
    notebook["cells"][11] = code(train)
    reporting = "".join(notebook["cells"][12]["source"])
    start = reporting.index('    qc_rep = run_eval_or_qc(')
    end = reporting.index('\nif sub_p.is_file()', start)
    block = reporting[start:end]
    reporting = reporting[:start] + '    if RUN_LOCAL_TRAIN_QC:\n' + textwrap.indent(block, '    ') + reporting[end:]
    notebook["cells"][12] = code(reporting)
    download = "".join(notebook["cells"][13]["source"])
    download = download.replace('    OUTPUTS_DIR / "submission.zip",',
                                '    OUTPUTS_DIR / "submission.zip",\n    OUTPUTS_DIR / "recovery_main_submission.zip",')
    notebook["cells"][13] = code(download)
    intro = "".join(notebook["cells"][0]["source"])
    intro = '''# V5 Balanced — cải thiện chất lượng trong ngân sách thời gian

Import file riêng lên Kaggle, bật GPU T4 và Internet. Không ghi đè notebook V4/V5 Score Focus.
**Recipe chưa train: chưa xác nhận mức tăng điểm hoặc thời gian chạy thật.** Mục tiêu dự trù khoảng 5–8 giờ trên một T4 khi dữ liệu tải/chuyển đổi bình thường; không phải cam kết SLA.

## Giảm công việc, giữ kiểm tra chất lượng
- Giữ sửa SAM2 BGR, teacher consensus hai góc nhìn, cách ly bất đồng và Golden Val đầy đủ nhãn.
- Bỏ COCO trong profile mặc định để tránh tải nguyên bộ ~25GB và vòng bổ sung nhãn; ưu tiên Mapillary 2800 + BDD 600 + khoảng 900 ảnh VN trước lọc/holdout. Đây là tradeoff dữ liệu cần đo, không khẳng định bỏ COCO sẽ tăng điểm.
- Teacher 15 epochs (mask_ratio=4), train chính 65 epochs (batch16, mask_ratio=4), fine-tune tối đa 8 epochs trên ảnh nhãn đầy đủ (batch8, mask_ratio=2). Các ảnh lớp yếu vẫn được repeat/crop; giới hạn repeat bổ sung 20%.
- Ngân sách 7,5 giờ tính từ Cell 1; callback dừng train theo batch, chừa 45 phút cho validation/export. Đây không phải giới hạn cứng cho tải mạng/giải nén/QC và vẫn có thể vượt khi các bước đó chậm.
- Lưu checkpoint mỗi 5 epochs; xuất recovery_main_submission.zip đã qua validator trước fine-tune. Nếu không đủ giờ, bỏ fine-tune; nếu fine-tune giảm official score thì giữ main.
- Vẫn chấm chính thức toàn Golden Val. Chỉ hoãn bước chẩn đoán QC suy luận 500 ảnh train (`RUN_LOCAL_TRAIN_QC=False`).
- Không dùng `time=` của trainer vì chế độ đó có thể tính lại số epochs/lịch LR. Callback giữ lịch epochs cấu hình, yêu cầu kết thúc train và lưu checkpoint bình thường.

Chạy tuần tự các cell; chỉ chạy lại Cell 1 khi thực sự bắt đầu phiên mới vì nó đặt lại đồng hồ. Nếu chuẩn bị dữ liệu chiếm gần hết giờ, notebook dừng với hướng dẫn giữ staged data; không xuất model chưa train. Checkpoint trong runs_v5_balanced/runs_stage1_v5_balanced có thể dùng để phục hồi khi phiên bị ngắt; cần giữ cùng dữ liệu/cấu hình khi resume.

''' + intro[intro.index('## Cơ sở nghiên cứu'):]
    intro += '\n- [Ultralytics 8.4.214 trainer callbacks/stop](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.214/ultralytics/engine/trainer.py): kết thúc batch, validate và lưu checkpoint khi stop được đặt.\n'
    notebook["cells"][0]["source"] = intro.splitlines(keepends=True)
    notebook["metadata"]["score_focus_v5"]["variant"] = "balanced_untrained"
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = f"v5b-{index:02d}-" + hashlib.sha256("".join(cell["source"]).encode()).hexdigest()[:8]
    return notebook


def verify(notebook):
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"balanced_cell_{i}", "exec")
    bundle = unpack_bundle(notebook)[2]
    for name, source in bundle.items():
        if name.endswith(".py"):
            compile(source, name, "exec")
    assert len(bundle) == 20


if __name__ == "__main__":
    notebook = build_notebook()
    verify(notebook)
    OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Created {OUTPUT.name}; 14 cells, 20 embedded files; original notebooks untouched.")
