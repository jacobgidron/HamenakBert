import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from dataset import get_xy, load_data

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
        self.linear = nn.Linear(768, NIKUD_NUM + DAGESH_NUM + SHIN_NUM)

    def forward(self, x):
        return self.linear(self.model(x)['last_hidden_state'])


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
model = MenakBert()
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
#
tokenizer = AutoTokenizer.from_pretrained("tau/tavbert-he")

train_dict = {}
train_dict["test"] = get_xy(load_data(tuple(['test1.txt']), maxlen=64).shuffle())
train_dict["train"] = get_xy(load_data(tuple(['train1.txt']), maxlen=64).shuffle())
import torch.optim as optim


def Criterion(n, s, d):
    return nn.CrossEntropyLoss(n) + nn.CrossEntropyLoss(d) + nn.CrossEntropyLoss(s)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i in range(len(train_dict["train"][0])):
        # get the inputs; data is a list of [inputs, labels]
        inputs = train_dict["train"][0][i]
        labels_D = train_dict["test"][1]["D"][i]
        labels_N = train_dict["test"][1]["N"][i]
        labels_S = train_dict["test"][1]["S"][i]

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(tokenizer.encode(inputs,return_tensors="pt"))
        loss = criterion(outputs, labels_D)+criterion(outputs, labels_N)+criterion(outputs, labels_S)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')
