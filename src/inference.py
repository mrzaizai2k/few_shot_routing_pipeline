import sys
import time

sys.path.append("")

from setfit import SetFitModel

# -----------------------------
# Timing: model loading
# -----------------------------
t0 = time.perf_counter()
model = SetFitModel.from_pretrained(
    "models/0226_15_59_routing_model",
    device="cuda"
)
t_load = time.perf_counter() - t0

# -----------------------------
# Input texts
# -----------------------------
texts = [
    "Please open the trunk and play some music for me",
    "Vui lòng mở cốp xe",
    "トランクを開けてください",
    "트렁크를 열어 주세요",

    "How does cruise control work",
    "Hệ thống ga tự động hoạt động thế nào",
    "クルーズコントロールはどう機能しますか",
    "크루즈 컨트롤은 어떻게 작동하나요",

    "Navigate to the nearest gas station",
    "Chỉ đường đến trạm xăng gần nhất",
    "最寄りのガソリンスタンドへ案内して",
    "가장 가까운 주유소로 안내해 주세요",

    "Play some relaxing music",
    "Phát một bản nhạc thư giãn",
    "リラックスできる音楽を再生して",
    "편안한 음악을 재생해 주세요",

    "Check for software updates",
    "Kiểm tra cập nhật phần mềm",
    "ソフト웨어の更新を確認して",
]

# -----------------------------
# Timing: predict
# -----------------------------
t1 = time.perf_counter()
preds = model.predict(texts)
t_predict = time.perf_counter() - t1

# -----------------------------
# Timing: predict_proba
# -----------------------------
t2 = time.perf_counter()
probs = model.predict_proba(texts)
t_proba = time.perf_counter() - t2

# -----------------------------
# Output
# -----------------------------
print("Predictions:", preds)
print("Probabilities shape:", probs.shape)

print("\nTiming results:")
print(f"Model load time      : {t_load:.4f} s")
print(f"Predict time         : {t_predict:.4f} s")
print(f"Predict_proba time   : {t_proba:.4f} s")
print(f"Total inference time : {(t_predict + t_proba):.4f} s")
print(f"Avg time / sentence  : {(t_predict + t_proba) / len(texts):.6f} s")