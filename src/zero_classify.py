import time
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    device=0 , # remove if CPU,
    # batch_size=16,
)


candidate_labels = [
    "open left door", "close left door", 
    "open right door", "close right door",
    "open window (front left)", "close window (front left)",
    "open window (front right)", "close window (front right)",
    "open window (rear left)", "close window (rear left)",
    "open window (rear right)", "close window (rear right)",
    "turn up air condition", "turn down air condition",
    "open trunk", "close trunk",
    "open sunroof", "close sunroof", "unknown_label", 
]

# --------------------------------------------------
# 1️⃣ WARM-UP (dummy batch)
# --------------------------------------------------
warmup_sequences = [
    "dummy sentence",
    "this is a warm up",
    "モデルをウォームアップします"
]

_ = classifier(
    warmup_sequences,
    candidate_labels,
    multi_label=True
)

# --------------------------------------------------
# 2️⃣ REAL BATCH
# --------------------------------------------------
# sequences = [
#     "dummy sentence",
#     "Tao muốn mở quyển sách lên để đọc.",
#     "open the doors and turn up air condition",
#     "Mở cửa sổ bên ghế lái",
#     "ドアを開けてエアコンを効かせましょう"
# ]

sequences = [
    "open the door and close the window", 
    "play music,open the front left window and turn down the air condition and tell me info of tesla",
    # "close the rear right window and open the trunk",
    # "open the sunroof and turn up the air condition",
    # "close the left door and close the rear left window",
    # "open the right door, open the trunk, and turn up the air condition",
    "tell me a joke, by the way tell me info of tesla turn down the air condition and close the sunroof"
]

threshold = 0.2

start_time = time.time()
results = classifier(sequences, candidate_labels, multi_label=True)
end_time = time.time()

# --------------------------------------------------
# 3️⃣ PRINT ONLY CONFIDENT RESULTS
# --------------------------------------------------
for seq, res in zip(sequences, results):
    print("=" * 60)
    print("Input:", seq)

    hits = False
    for label, score in zip(res["labels"], res["scores"]):
        if score >= threshold:
            print(f"{label:30s} {score:.4f}")
            hits = True

    if not hits:
        print("NO LABEL ≥ 0.5")

print(f"\nExecution time (after warm-up): {end_time - start_time:.3f}s")

