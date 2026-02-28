
import sys
sys.path.append("")

from datetime import datetime
import os
from datasets import load_dataset, ClassLabel
from setfit import SetFitModel, Trainer, TrainingArguments
from src.utils import config_parser

config = config_parser(data_config_path = 'config/routing_config.yaml')
dataset_name = config['dataset_name']
sentence_model = config['sentence_model']
text_col_name = config['text_col_name']
label_col_name = config['label_col_name']

# validation_split = 25

datasets = load_dataset(
    "csv",
    data_files=config['local_data_path'],
    split="train"
)

# datasets = load_dataset(dataset_name, split="train", cache_dir=True)

# Assuming the loaded dataset is named "datasets"
new_features = datasets.features.copy()  # Get dataset features
new_features[label_col_name] = ClassLabel(num_classes = len(datasets.unique(label_col_name)), names=datasets.unique(label_col_name))  # Extract unique labels and cast
datasets = datasets.cast(new_features)
ddict  = datasets.train_test_split(test_size=config['test_size'], stratify_by_column=label_col_name, shuffle =True)
train_ds, val_ds = ddict["train"], ddict["test"]
ddict_val_test  = val_ds.train_test_split(test_size=0.5, stratify_by_column=label_col_name, shuffle =True)
val_ds, test_ds = ddict_val_test["train"], ddict_val_test["test"]

# Load a SetFit model from Hub
model = SetFitModel.from_pretrained(
    sentence_model,
    labels=datasets.unique(label_col_name),
)

args = TrainingArguments(
    batch_size=config['batch_size'],
    num_epochs=config['num_epochs'],
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

args.eval_strategy = args.evaluation_strategy # SetFitpackage error

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    metric="accuracy",
    column_mapping={"text": text_col_name, "label": label_col_name}  # Map dataset columns to text/label expected by trainer
)

trainer.train()

metrics = trainer.evaluate(test_ds)
print("metrics", metrics)

preds = model.predict([
    # Class 0 – Car Control
    "Please open the trunk",                 # en
    "Vui lòng mở cốp xe",                     # vi
    "トランクを開けてください",                # jp
    "트렁크를 열어 주세요",                    # ko

    # Class 1 – Car Manual
    "How does cruise control work",           # en
    "Hệ thống ga tự động hoạt động thế nào",   # vi
    "クルーズコントロールはどう機能しますか", # jp
    "크루즈 컨트롤은 어떻게 작동하나요",      # ko

    # Class 2 – Navigation
    "Navigate to the nearest gas station",    # en
    "Chỉ đường đến trạm xăng gần nhất",        # vi
    "最寄りのガソリンスタンドへ案内して",      # jp
    "가장 가까운 주유소로 안내해 주세요",      # ko

    # Class 3 – Infotainment
    "Play some relaxing music",               # en
    "Phát một bản nhạc thư giãn",              # vi
    "リラックスできる音楽を再生して",          # jp
    "편안한 음악을 재생해 주세요"              # ko

    # Class 4 – Cloud
    "Check for software updates",              # en
    "Kiểm tra cập nhật phần mềm",               # vi
    "ソフトウェアの更新を確認して",              # jp
    "소프트웨어 업데이트를 확인해 주세요"        # ko
])

print("Prediction:", preds)


if config.get("push_to_hub"):
    try:
        trainer.push_to_hub(config["huggingface_out_dir"])
        print(f"Pushed model to Hugging Face Hub: {config['huggingface_out_dir']}")
    except Exception as e:
        print("❌ Failed to push model to Hub:", e)


if config.get("model_output_path"):
    try:
        timestamp = datetime.now().strftime("%m%d_%H_%M")
        save_dir = os.path.join(
            config["model_output_path"],
            f"{timestamp}_routing_model"
        )

        os.makedirs(save_dir, exist_ok=True)

        model.save_pretrained(save_dir)
        print(f"✅ Model saved locally at: {save_dir}")

    except Exception as e:
        print("❌ Failed to save model locally:", e)



