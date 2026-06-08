import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#Loading data from CSV files

train_df = pd.read_csv("data/hallucination_train_1.csv")
test_df  = pd.read_csv("data/hallucination_test_1.csv")

#Dropping rows with missing values and resetting index

train_df = train_df.dropna(
    subset=["question", "context", "answer", "label"]
).reset_index(drop=True)

test_df = test_df.dropna(
    subset=["question", "context", "answer", "label"]
).reset_index(drop=True)

train_df["label"] = train_df["label"].astype(int)
test_df["label"]  = test_df["label"].astype(int)

print("Train label distribution:\n", train_df["label"].value_counts())
print("Test  label distribution:\n", test_df["label"].value_counts())

# Converting pandas DataFrames to Hugging Face Datasets

train_ds = Dataset.from_pandas(train_df)
test_ds  = Dataset.from_pandas(test_df)

# Loading the tokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

# Preprocessing function to tokenize the inputs and prepare labels

def preprocess(example):
    text = str(example["question"]) + " [SEP] "  + str(example["answer"])

    encoding = tokenizer(
        text,
        str(example["context"]),
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    encoding["labels"] = int(example["label"])  

    return encoding

cols_to_remove = ["question", "context", "answer", "label"]

train_ds = train_ds.map(preprocess, remove_columns=cols_to_remove)
test_ds  = test_ds.map(preprocess,  remove_columns=cols_to_remove)

# Sanity check
print("\nSample after preprocessing:")
print({k: v for k, v in train_ds[0].items() if k != "input_ids"})  
assert "labels"    in train_ds.column_names, "labels column missing!"
assert "label"     not in train_ds.column_names, "old label column still present!"
assert "input_ids" in train_ds.column_names, "input_ids missing!"

# Loading the pre-trained model

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=2
)

# Setting up training arguments

args = TrainingArguments(
    output_dir="detector",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    load_best_model_at_end=True,
    disable_tqdm=False
)

# Setting up metrics for evaluation

accuracy_metric = evaluate.load("accuracy")
f1_metric       = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_metric.compute(
            predictions=predictions, references=labels
        )["accuracy"],
        "f1": f1_metric.compute(
            predictions=predictions, references=labels
        )["f1"]
    }

# Setting up trainer

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

# Training the model

trainer.train()

# Evaluating the model on the test set and printing results

results = trainer.evaluate()
print("\nFinal evaluation results:")
print(results)

predictions = trainer.predict(test_ds)

preds = np.argmax(
    predictions.predictions,
    axis=1
)

labels = predictions.label_ids

cm = confusion_matrix(labels, preds)

sns.heatmap(
    cm,
    annot=True,
    fmt="d"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print(
    classification_report(
        labels,
        preds
    )
)

# Saving the model and tokenizer

save_path = "hallucination_detector"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print(f"\nModel and tokenizer saved to '{save_path}/'")