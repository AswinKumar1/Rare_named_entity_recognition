# https://huggingface.co/datasets/wnut_17   (Dataset Link)

!pip install accelerate>=0.20.1

!pip install transformers[torch]

!pip install datasets

!pip install -q evaluate seqeval

from datasets import load_dataset
wnut = load_dataset("wnut_17")
wnut

print(wnut["train"][0])

label_list = wnut["train"].features[f"ner_tags"].feature.names
label_list

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

example = wnut["train"][0]
tokenized_input = tokenizer(example["tokens"], is_split_into_words = True)
tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
print(tokens)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_wnut = wnut.map( tokenize_and_align_labels, batched = True)

## padding
from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer = tokenizer)

#!pip install -q evaluate seqeval

import evaluate
seqeval = evaluate.load("seqeval")

import numpy as np

labels = [label_list[i] for i in example[f"ner_tags"]]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
}

id2label = {
    0: "O",
    1: "B-corporation",
    2: "I-corporation",
    3: "B-creative-work",
    4: "I-creative-work",
    5: "B-group",
    6: "I-group",
    7: "B-location",
    8: "I-location",
    9: "B-person",
    10: "I-person",
    11: "B-product",
    12: "I-product",
    }
label2id = {
    "O": 0,
    "B-corporation": 1,
    "I-corporation": 2,
    "B-creative-work": 3,
    "I-creative-work": 4,
    "B-group": 5,
    "I-group": 6,
    "B-location": 7,
    "I-location": 8,
    "B-person": 9,
    "I-person": 10,
    "B-product": 11,
    "I-product": 12,
}

from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels = 13,
    id2label = id2label,
    label2id= label2id
)

#!pip install transformers[torch]

#!pip install accelerate>=0.20.1

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir = "my_ner_model",
    learning_rate = 2e-5, # we can change it
    per_device_train_batch_size = 1, # we can change it
    per_device_eval_batch_size=1, # we can change it
    num_train_epochs = 2,   # we can change it
    weight_decay = 0.01, # we can change it
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,


)

from transformers import Trainer

trainer = Trainer(
    model = model,
    train_dataset = tokenized_wnut["train"],
    eval_dataset = tokenized_wnut["test"],
    tokenizer = tokenizer,
    data_collator = data_collator,
    compute_metrics = compute_metrics,
    args = training_args
)

trainer.train()

## Inference ( optional: You can remove 3 cells coming as follows:)

from transformers import pipeline

text = "My name is Sarah, I live in London and New York with Obama"
classifier = pipeline("ner", model="/content/my_ner_model/checkpoint-6788")
classifier(text)

import pandas as pd

def tag_sentence(text:str):
    # convert our text to a  tokenized sequence
    inputs = tokenizer(text, truncation=True, return_tensors="pt").to("cuda")
    # get outputs
    outputs = model(**inputs)
    # convert to probabilities with softmax
    probs = outputs[0][0].softmax(1)
    # get the tags with the highest probability
    word_tags = [(tokenizer.decode(inputs['input_ids'][0][i].item()), id2label[tagid.item()])
                  for i, tagid in enumerate (probs.argmax(axis=1))]

    return pd.DataFrame(word_tags, columns=['word', 'tag'])

print(tag_sentence(text))
