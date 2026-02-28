
import sys
sys.path.append("")

from setfit import SetFitModel

# Download from Hub
model = SetFitModel.from_pretrained("models/0226_15_59_routing_model")
# Run inference
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

# Predicted class (argmax)
preds = model.predict(texts)


probs = model.predict_proba(texts)

print("Predictions:", preds)
print("Probabilities:", probs)
print("Probabilities shape:", probs.shape)