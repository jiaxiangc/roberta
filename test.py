from transformers import AutoTokenizer, RobertaForMultipleChoice
import torch

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = RobertaForMultipleChoice.from_pretrained("roberta-base")

prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
question1 = "Who is cool"
choice0 = "It is eaten with a fork and a knife."
choice1 = "It is eaten while held in the hand."
# choice2 = "It is eaten while held in the hand."
# choice3 = "It is eaten while held in the hand."
labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

encoding = tokenizer([(prompt + question1) for _ in range(3)], [choice0, choice1, choice0], return_tensors="pt", padding=True)
# encoding = tokenizer.encode_plus([prompt, prompt], [choice0, choice1], return_tensors="pt", padding=True)

# print("====", encoding)
input_ids = encoding['input_ids'].unsqueeze(0)
attention_mask = encoding['attention_mask'].unsqueeze(0)
print(input_ids.shape)

outputs = model(input_ids=input_ids, attention_mask=attention_mask)  # batch size is 1

# the linear classifier still needs to be trained
# loss = outputs.loss
logits = outputs.logits
print(logits.shape)
# criterion = torch.nn.CrossEntropyLoss()
# logits = torch.zeros(size=(1, 3)).float()
# logits[0][0] = 1.0
# print(logits)
# labels = torch.tensor([[1, 0, 0]]).float()
# print(labels)
# loss = criterion(logits, labels)
# print(loss)