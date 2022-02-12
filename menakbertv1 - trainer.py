import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from dataset import textDataset, NIQQUD_SIZE, DAGESH_SIZE, SIN_SIZE
from torch import nn

MAX_LEN = 100
tokenizer = AutoTokenizer.from_pretrained("tau/tavbert-he")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MenakBert(torch.nn.Module):
    """
    the model
    """

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("tau/tavbert-he")
        self.linear_D = nn.Linear(768, DAGESH_SIZE)
        self.linear_S = nn.Linear(768, SIN_SIZE)
        self.linear_N = nn.Linear(768, NIQQUD_SIZE)

    def forward(self, x, y1):
        return self.linear_N(self.model(x)['last_hidden_state']) \
            , self.linear_D(self.model(x)['last_hidden_state']) \
            , self.linear_S(self.model(x)['last_hidden_state'])


# def tokenize_function(example):
#     """
#     the tokenize function get a single sentence
#     and returns a tensor of the character sentece
#
#     """
#
#
# sentences = [x.lower() for x in example['text']]
# tokenized_sentences = [word_tokenize(x) for x in sentences]
# tokenized_idx = [[vocab[word] if word in vocab else vocab["unk"] for word in x] for x in tokenized_sentences]
# max_size = max([len(x) for x in tokenized_idx])
# final_tokenized_idx = tokenized_idx
# return {"labels": example['label'], 'input_ids': final_tokenized_idx}
#


# from transformers import Trainer
# from transformers import TrainingArguments
#
# # co = DataCollatorWithPadding()
# training_args = TrainingArguments("MenakBert",
#                                   # YOUR CODE HERE
#                                   num_train_epochs=40,  # must be at least 10.
#                                   per_device_train_batch_size=100,
#                                   per_device_eval_batch_size=100,
#                                   learning_rate=0.01,
#                                   # END YOUR END
#
#                                   save_total_limit=2,
#                                   log_level="error",
#                                   evaluation_strategy="epoch")
#

from dataclasses import dataclass


@dataclass
class DataCollatorWithPadding:

    def __call__(self, features):
        batch = tokenizer([x.get("text") for x in features], padding='max_length', max_length=MAX_LEN,
                          return_tensors="pt")
        features_dict = {}
        features_dict["y1"] = {
            "N": torch.tensor([x.get("y1").get("N") for x in features]).long(),
            "D": torch.tensor([x.get("y1").get("D") for x in features]).long(),
            "S": torch.tensor([x.get("y1").get("S") for x in features]).long(),
        }
        features_dict["x"] = batch.data["input_ids"]
        # features_dict["tokens"] = [tokenizer.encode(x.get("text"),return_tensors="pt") for x in features]

        # features_dict["input_ids"] = torch.tensor([pad_sequence_to_length(x, max_len) for x in input_ids]).long()
        # features_dict["attention_masks"] = torch.tensor([pad_sequence_to_length(x, max_len) for x in masks]).long()

        return features_dict


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("y1")
        # forward pass
        outputs = model(**inputs)
        # outputs = [model(**inputs, x=x) for x in inputs.get("tokens")]
        logits = outputs
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits[0].permute((0, 2, 1)), labels["N"]) + loss_fct(logits[1].permute((0, 2, 1)),
                                                                              labels["D"]) + loss_fct(
            logits[2].permute((0, 2, 1)), labels["D"])
        return (loss, outputs) if return_outputs else loss


model = MenakBert()

training_args = TrainingArguments("MenakBert",
                                  num_train_epochs=40,
                                  per_device_train_batch_size=10,
                                  per_device_eval_batch_size=100,
                                  learning_rate=0.01,
                                  save_total_limit=2,
                                  log_level="error",
                                  evaluation_strategy="epoch")

# from datasets import load_metric
#
# metric = load_metric("accuracy")
#
# def compute_metrics(eval_pred):
#     logits = eval_pred.predictions
#     labels = eval_pred.label_ids
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
small_train_dataset = textDataset(tuple(['train1.txt']), MAX_LEN - 1)
small_eval_dataset = textDataset(tuple(['train1.txt']), MAX_LEN - 1)

co = DataCollatorWithPadding()

trainer = CustomTrainer(
    model=model,
    data_collator=co,
    args=training_args,
    callbacks=[
        # YOUR CODE HERE
        # END YOUR END
    ],
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    # compute_metrics=compute_metrics,
)
trainer.train()
