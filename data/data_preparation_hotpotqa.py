from datasets import load_dataset
import pandas as pd
from rank_bm25 import BM25Okapi

# Load the HotpotQA dataset with the "distractor" configuration

dataset = load_dataset(
    "hotpotqa/hotpot_qa",
    "distractor"
)

# Converting the train and test splits to pandas DataFrames and shuffling them

df_trn = pd.DataFrame(dataset["train"])
df_tst = pd.DataFrame(dataset["validation"])
df_trn = df_trn.sample(frac=1, random_state=42).reset_index(drop=True)
df_tst = df_tst.sample(frac=1, random_state=42).reset_index(drop=True)

# Function to extract supporting facts from the context

def extract_supporting_facts(example):

    titles = example["context"]["title"]
    sentences = example["context"]["sentences"]

    title_to_sentences = {
        title: sents
        for title, sents in zip(titles, sentences)
    }

    evidence = []

    for title, sent_idx in zip(
        example["supporting_facts"]["title"],
        example["supporting_facts"]["sent_id"]
    ):

        if title in title_to_sentences:

            sents = title_to_sentences[title]

            if sent_idx < len(sents):
                evidence.append(
                    sents[sent_idx]
                )

    return " ".join(evidence)

#Creating positive examples


def create_positive(example):


    return [
        example["question"],
        extract_supporting_facts(example),
        example["answer"],
        0
    ]


#Creating negative examples

def create_negative(
    example,
    dataset,
    bm25
):

    query = example["question"]

    scores = bm25.get_scores(
        query.lower().split()
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )

    correct_answer = example["answer"]

    for idx in ranked_indices:

        candidate = dataset.iloc[idx]

        if candidate["question"] == query:
            continue

        candidate_answer = candidate["answer"]

        if candidate_answer == correct_answer:
            continue

        return [
            example["question"],
            extract_supporting_facts(example),
            candidate_answer,
            1
        ]



#Training examples

train_questions = df_trn["question"].tolist()

tokenized_train_questions = [
    q.lower().split()
    for q in train_questions
]

train_bm25 = BM25Okapi(
    tokenized_train_questions
)

print("Creating training examples...")

df_trn_new = pd.DataFrame(columns=["question", "evidence", "answer", "label"])
count=0
i = 0
while count < 16000:
    pos_example = create_positive(df_trn.iloc[i])
    neg_example = create_negative(df_trn.iloc[i], df_trn, train_bm25)
    df_trn_new.loc[len(df_trn_new)] = pos_example
    count+=1
    if count % 100 == 0:
        print(count,'/',16000)
    if neg_example is not None:
        df_trn_new.loc[len(df_trn_new)] = neg_example
        count+=1
        if count % 100 == 0:
            print(count,'/',16000)
    i+=1

#Testing examples

test_questions = df_tst["question"].tolist()

tokenized_test_questions = [
    q.lower().split()
    for q in test_questions
]

test_bm25 = BM25Okapi(
    tokenized_test_questions
)

print("\nCreating test examples...") 

df_tst_new = pd.DataFrame(columns=["question", "evidence", "answer", "label"])
count=0
i = 0
while count < 4000:
    pos_example = create_positive(df_tst.iloc[i])
    neg_example = create_negative(df_tst.iloc[i], df_tst, test_bm25)
    df_tst_new.loc[len(df_tst_new)] = pos_example
    count+=1
    if count % 100 == 0:
        print(count,'/',4000)
    if neg_example is not None:
        df_tst_new.loc[len(df_tst_new)] = neg_example
        count+=1
        if count % 100 == 0:
            print(count,'/',4000)
    i+=1

# Shuffling the dataframes and resetting the index

df_trn_new = df_trn_new.sample(frac=1, random_state=42).reset_index(drop=True)
df_tst_new = df_tst_new.sample(frac=1, random_state=42).reset_index(drop=True)


# Printing the sizes of the new dataframes, checking for duplicates and printing label distributions

print("\nTrain size:", len(df_trn_new))
print("Test size:", len(df_tst_new))

duplicates = df_trn_new.duplicated(
    subset=["question", "evidence", "answer", "label"]
).sum()

print("\nTrain duplicates:", duplicates)

duplicates = df_tst_new.duplicated(
    subset=["question", "evidence", "answer", "label"]
).sum()

print("Test duplicates:", duplicates)

print("\nTrain labels:")
print(df_trn_new["label"].value_counts())

print("\nTest labels:")
print(df_tst_new["label"].value_counts())

# Saving the new dataframes to CSV files

df_trn_new.to_csv(
    "data/hallucination_train_1.csv",
    index=False
)

df_tst_new.to_csv(
    "data/hallucination_test_1.csv",
    index=False
)