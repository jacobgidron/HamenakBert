import torch
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from dataset import textDataset, NIQQUD_SIZE, DAGESH_SIZE, SIN_SIZE ,Y2_SIZE
from torch import nn
import numpy as np
import sklearn


# import sklearn
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
        self.linear = nn.Linear(768, Y2_SIZE)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y2=None ,**kwargs):
        res = self.linear(self.model(x)['last_hidden_state'])
        loss = self.loss(res.permute((0,2,1)),y2)
        return {"loss":loss,"logits":res}


from datasets import load_metric

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions.flatten(), references=labels.flatten())

from dataclasses import dataclass


@dataclass
class DataCollatorWithPadding:

    def __call__(self, features):
        batch = tokenizer([x.get("text") for x in features], padding='max_length', max_length=MAX_LEN,
                          return_tensors="pt")
        features_dict = {}
        features_dict["y2"] = torch.tensor([x.get("y2") for x in features]).long()
        features_dict["labels"] = torch.tensor([x.get("y2") for x in features]).long()

        features_dict["x"] = batch.data["input_ids"]
        # features_dict["tokens"] = [tokenizer.encode(x.get("text"),return_tensors="pt") for x in features]

        # features_dict["input_ids"] = torch.tensor([pad_sequence_to_length(x, max_len) for x in input_ids]).long()
        # features_dict["attention_masks"] = torch.tensor([pad_sequence_to_length(x, max_len) for x in masks]).long()

        return features_dict

loss_fct1 = nn.CrossEntropyLoss()
loss_fct2 = nn.CrossEntropyLoss()
loss_fct3 = nn.CrossEntropyLoss()
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("y1")
        # forward pass
        outputs = model(**inputs)
        # outputs = [model(**inputs, x=x) for x in inputs.get("tokens")]
        logits = outputs
        # compute custom loss (suppose one has 3 labels with different weights)

        loss = loss_fct1(logits[0].permute((0, 2, 1)), labels["N"]) + \
               loss_fct2(logits[1].permute((0, 2, 1)),
                         labels["D"]) + loss_fct3(logits[2].permute((0, 2, 1)), labels["D"])
        return (loss, outputs[2]) if return_outputs else loss


model = MenakBert()

training_args = TrainingArguments("MenakBertv2",
                                  num_train_epochs=5,
                                  per_device_train_batch_size=10,
                                  per_device_eval_batch_size=10,
                                  learning_rate=0.05,
                                  logging_steps=64,
                                  save_total_limit=32,
                                  log_level="error",
                                  logging_dir="log",
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
small_eval_dataset = textDataset(tuple(['test1.txt']), MAX_LEN - 1)

co = DataCollatorWithPadding()

trainer = Trainer(
    model=model,
    data_collator=co,
    args=training_args,
    callbacks=[
        # YOUR CODE HERE
        # END YOUR END
    ],
    eval_dataset=small_train_dataset,
    train_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()
