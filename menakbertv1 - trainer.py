import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel, TrainingArguments , Trainer
from dataset import get_xy, load_data
from torch import nn

NIKUD_NUM = 16
SHIN_NUM = 3
DAGESH_NUM = 2


class MenakBert(torch.nn.Module):
    """
    the model
    """

    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained("tau/tavbert-he")
        self.linear_D = nn.Linear(768, DAGESH_NUM)
        self.linear_S = nn.Linear(768, SHIN_NUM)
        self.linear_N = nn.Linear(768, NIKUD_NUM)

    def forward(self, x):
        return self.linear_N(self.model(x)['last_hidden_state'])\
            ,self.linear_D(self.model(x)['last_hidden_state'])\
            ,self.linear_S(self.model(x)['last_hidden_state'])


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


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits[0], labels["N"])+loss_fct(logits[1], labels["D"])+loss_fct(logits[2], labels["S"])
        return (loss, outputs) if return_outputs else loss





model = MenakBert()

training_args = TrainingArguments("MenakBert",
                                  num_train_epochs=40,
                                  per_device_train_batch_size=100,
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

trainer = Trainer(
    model=model,
    # data_collator=co,
    args=training_args,
    callbacks=[
        # YOUR CODE HERE
        # END YOUR END
    ],
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()

tokenizer = AutoTokenizer.from_pretrained("tau/tavbert-he")
train_dict = {}
train_dict["test"] = get_xy(load_data(tuple(['test1.txt']), maxlen=64).shuffle())
train_dict["train"] = get_xy(load_data(tuple(['train1.txt']), maxlen=64).shuffle())