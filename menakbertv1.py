import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

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
        self.classifier = nn.Sequential(
            self.model,
            nn.Linear(768, NIKUD_NUM + DAGESH_NUM + SHIN_NUM)
        )

    def forward(self, x):
        return self.classifier(x)




def tokenize_function(example):
    """
    the tokenize function get a single sentence
    and returns a tensor of the character sentece

    """
  sentences = [x.lower() for x in example['text']]
  tokenized_sentences = [word_tokenize(x) for x in sentences]
  tokenized_idx = [[vocab[word] if word in vocab else vocab["unk"] for word in x] for x in tokenized_sentences]
  max_size = max([len(x) for x in tokenized_idx])
  final_tokenized_idx = tokenized_idx

  return {"labels":example['label'],'input_ids':final_tokenized_idx}

from transformers import Trainer
from transformers import TrainingArguments

# co = DataCollatorWithPadding()
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
# model = MenakBert()
#
# trainer = Trainer(
#     model=model,
#     # data_collator=co,
#     args=training_args,
#     callbacks=[
#         # YOUR CODE HERE
#         # END YOUR END
#     ],
#     train_dataset=small_train_dataset,
#     eval_dataset=small_eval_dataset,
#     compute_metrics=compute_metrics,
# )
# trainer.train()