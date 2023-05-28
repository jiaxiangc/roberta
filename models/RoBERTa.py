import torch.nn as nn
from transformers import RobertaForMultipleChoice


class RoBERTaClassifier(nn.Module):
    def __init__(self):
        super(RoBERTaClassifier, self).__init__()
        self.roberta = RobertaForMultipleChoice.from_pretrained('roberta-base')

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # TODO: what symbol
        logits = outputs.logits
        return logits