import json
import re
import os

import torch
import transformers
transformers.logging.set_verbosity_error()

from transformers import RobertaTokenizer

from models.RoBERTa import RoBERTaClassifier


def main():
    dic = {0: "A", 1: "B", 2: "C", 3: "D"}
    # 从一个文件中加载：1 article，1 question，4 answer
    device = torch.device('cuda:3')
    checkpinits = 'checkpoints/roberta_race.pt'

    res = []
    file_path = './RACE/dev/middle/13.txt'

    print('Loading RoBERTa Tokenizer')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RoBERTaClassifier().to(device)
    # model.load_state_dict(torch.load(checkpinits))

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        data = json.loads(content)

        article = data['article'].replace('\n', ' ')
        article = re.sub(r"\s+", " ", article)

        questions = data['questions']
        options = data['options']
        answers = data['answers']
        for i in range(len(questions)):
            qa_list = []
            question = questions[i]
            for j in range(4):
                option = options[i][j]
                if "_" in question:
                    qa_cat = question.replace("_", option)
                else:
                    qa_cat = " ".join([question, option])
                qa_cat = re.sub(r"\s+", " ", qa_cat)
                qa_list.append(qa_cat)


            res.append((article, qa_list, answers[i]))

    for d in res:
        encoded_inputs = tokenizer(
            [d[0] for _ in range(len(d[1]))],
            d[1],
            max_length=512,
            padding='max_length',
            truncation=True,
            # TODO: what
            return_tensors='pt'
        )
        input_ids = encoded_inputs['input_ids'].unsqueeze(0).to(device)
        attention_mask = encoded_inputs['attention_mask'].unsqueeze(0).to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        print('logits: ', logits)
        print('predict answer: ', dic[torch.max(logits, dim=1).indices.item()])



if __name__ == '__main__':
    main()