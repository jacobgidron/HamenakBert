import torch
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from dataset import textDataset, NIQQUD_SIZE, DAGESH_SIZE, SIN_SIZE
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
        self.linear_D = nn.Linear(768, DAGESH_SIZE)
        self.linear_S = nn.Linear(768, SIN_SIZE)
        self.linear_N = nn.Linear(768, NIQQUD_SIZE)

    def forward(self, x, y1, **kwargs):
        """

        :param x:
        :param y1:
        :return: tuple of 3 tensor batch,sentence_len,classes
        """
        last_hidden_state = self.model(x)['last_hidden_state']
        res = self.linear_N(last_hidden_state), \
              self.linear_D(last_hidden_state), \
              self.linear_S(last_hidden_state)
        return {"logits": res}


from datasets import load_metric

metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    print("hello")
    logits = eval_pred.predictions
    labels_N = eval_pred.label_ids.get("N")
    labels_D = eval_pred.label_ids.get("D")
    labels_S = eval_pred.label_ids.get("S")
    predictions_N = np.argmax(logits[0], axis=-1)
    predictions_D = np.argmax(logits[1], axis=-1)
    predictions_S = np.argmax(logits[2], axis=-1)
    acc = metric.compute(predictions=predictions_D, references=labels_D) + \
          metric.compute(predictions=predictions_N, references=labels_N) + \
          metric.compute(predictions=predictions_S, references=labels_S)
    return {"eval_loss": acc}


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
        features_dict["labels"] = features_dict["y1"]
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
        # model(x=inputs["x"],...)
        # outputs = [model(**inputs, x=x) for x in inputs.get("tokens")]
        logits = outputs['logits']
        # compute custom loss (suppose one has 3 labels with different weights)

        loss = loss_fct1(logits[0].permute((0, 2, 1)), labels["N"]) + \
               loss_fct2(logits[1].permute((0, 2, 1)), labels["D"]) + \
               loss_fct3(logits[2].permute((0, 2, 1)), labels["S"])
        return (loss, outputs) if return_outputs else loss


model = MenakBert()

training_args = TrainingArguments("MenakBert",
                                  num_train_epochs=4,
                                  per_device_train_batch_size=10,
                                  per_device_eval_batch_size=1,
                                  learning_rate=0.05,
                                  save_total_limit=2,
                                  log_level="error",
                                  logging_dir="log",
                                  eval_steps=20,
                                  evaluation_strategy="steps")

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

trainer = CustomTrainer(
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
