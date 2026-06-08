from datasets import load_dataset
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

nltk.download("punkt_tab")

#Loading SQuAD dataset and creating train and test dataframes

squad = load_dataset("squad")
df_trn = pd.DataFrame(squad["train"])
df_trn = df_trn.sample(frac=1, random_state=42).reset_index(drop=True)
df_tst = pd.DataFrame(squad["validation"])
df_tst = df_tst.sample(frac=1, random_state=42).reset_index(drop=True)

#Function to shorten context to fit within 512 tokens when combined with question and answer

def shorten_context(context, answer_text, window=1):
    """
    Returns the sentence containing the answer
    plus `window` sentences on each side.
    """

    sentences = sent_tokenize(context)

    answer_idx = None

    for i, sentence in enumerate(sentences):
        if answer_text in sentence:
            answer_idx = i
            break

    if answer_idx is None:
        return context

    start = max(0, answer_idx - window)
    end = min(len(sentences), answer_idx + window + 1)

    return " ".join(sentences[start:end])

# Checking if the combined length of question, context and answer exceeds 512 tokens

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/deberta-v3-base"
)

def exceeds_max_length(question, context, answer, max_length=512):

    text = (
        "Question: " + str(question) +
        " [SEP] Answer: " + str(answer) +
        " [SEP] Context: " + str(context)
    )

    tokenized = tokenizer(
        text,
        truncation=False
    )

    return len(tokenized["input_ids"]) > max_length

#Creating positive and negative examples for training and testing

def create_positive(example):

    answer = example["answers"]["text"][0]

    short_context = shorten_context(
        example["context"],
        answer,
        window=1
    )
    
    if exceeds_max_length(example["question"], short_context, answer):
        return -1

    return [
        example["question"],
        short_context,
        answer,
        0
    ]

def create_negative(example, example_df):
    df_context = example_df[example_df["context"] == example["context"]]
    candidate = df_context.sample(n=1).iloc[0]
    if len(df_context) == 1:
        return -1

    while candidate["id"] == example["id"]:
        candidate = df_context.sample(n=1).iloc[0]

    wrong_answer = candidate["answers"]["text"][0]

    answer = example["answers"]["text"][0]

    short_context = shorten_context(
        example["context"],
        answer,
        window=1
    )
    if exceeds_max_length(example["question"], short_context, wrong_answer):
        return -1
    return [
        example["question"],
        short_context,
        wrong_answer,
        1
    ]

print("Creating training examples...")

df_trn_new = pd.DataFrame(columns=["question", "context", "answer", "label"])
count=0
i = 0
while count < 16000:
    pos_example = create_positive(df_trn.iloc[i])
    neg_example = create_negative(df_trn.iloc[i], df_trn)
    if pos_example != -1:
        df_trn_new.loc[len(df_trn_new)] = pos_example
        count+=1
        if count % 1000 == 0:
            print(count,'/',16000)
    if neg_example != -1:
        df_trn_new.loc[len(df_trn_new)] = neg_example
        count+=1
        if count % 1000 == 0:
            print(count,'/',16000)
    i+=1

print("\nCreating test examples...")

df_tst_new = pd.DataFrame(columns=["question", "context", "answer", "label"])
count=0
i = 0
while count < 4000:
    pos_example = create_positive(df_tst.iloc[i])
    neg_example = create_negative(df_tst.iloc[i], df_tst)
    if pos_example != -1:
        df_tst_new.loc[len(df_tst_new)] = pos_example
        count+=1
        if count % 1000 == 0:
            print(count,'/',4000)
    if neg_example != -1:
        df_tst_new.loc[len(df_tst_new)] = neg_example
        count+=1
        if count % 1000 == 0:
            print(count,'/',4000)
    i+=1

# Shuffling the dataframes and resetting the index

df_trn_new = df_trn_new.sample(frac=1, random_state=42).reset_index(drop=True)
df_tst_new = df_tst_new.sample(frac=1, random_state=42).reset_index(drop=True)


# Printing the sizes of the new dataframes, checking for duplicates and printing label distributions

print("\nTrain size:", len(df_trn_new))
print("Test size:", len(df_tst_new))

duplicates = df_trn_new.duplicated(
    subset=["question", "context", "answer", "label"]
).sum()

print("\nTrain duplicates:", duplicates)

duplicates = df_tst_new.duplicated(
    subset=["question", "context", "answer", "label"]
).sum()

print("Test duplicates:", duplicates)

print("\nTrain labels:")
print(df_trn_new["label"].value_counts())

print("\nTest labels:")
print(df_tst_new["label"].value_counts())

# Checking the maximum length of the combined question, context and answer in the training set

list=[]
for i in range(15999):
    list.append(len(tokenizer(
        "Question: " + str(df_trn_new.iloc[i]["question"]) +
        " [SEP] Answer: " + str(df_trn_new.iloc[i]["answer"]) +
        " [SEP] Context: " + str(df_trn_new.iloc[i]["context"]),
        truncation=False
    )["input_ids"]))

print("\nMax length in train set:", max(list))

# Saving the new dataframes to CSV files

df_trn_new.to_csv(
    "data/hallucination_train_1.csv",
    index=False
)

df_tst_new.to_csv(
    "data/hallucination_test_1.csv",
    index=False
)
