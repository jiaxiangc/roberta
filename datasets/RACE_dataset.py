import json
import re
import os

import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import RobertaTokenizer

# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

class RACEDataset(Dataset):
    def __init__(self, data_dir):
        # 处理数据集，主要包含RoBERTa的分词器
        self.data = []
        print('Load RoBERTa tokenizer.')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        # _tqdm_idx = -1
        # print('Extract data from file of .txt.')
        for root, dirs, files in os.walk(data_dir):
            # _tqdm_idx += 1
            for file in tqdm(files, desc=f"Process dataset"):
                file_path = os.path.join(root, file)
                # self.file_paths.append(file_path)
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


                        self.data.append((article, qa_list, answers[i]))

                    
                    # for i in range(len(questions)):
                    #     encoded_inputs = self.tokenizer(
                    #         [article + ' ' + questions[i] for _ in range(len(options[i]))],
                    #         options[i],
                    #         # add_special_tokens=True,
                    #         max_length=512,
                    #         padding='max_length',
                    #         truncation=True,
                    #         # TODO: what
                    #         return_tensors='pt'
                    #     )
                    #     input_ids = encoded_inputs['input_ids']
                    #     attention_mask = encoded_inputs['attention_mask']
                    #     answer = torch.tensor(ord(answers[i]) - ord('A'))

                    #     self.data.append((input_ids, attention_mask, answer))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        encoded_inputs = self.tokenizer(
            [self.data[index][0] for _ in range(4)],
            self.data[index][1],
            max_length=512,
            padding='max_length',
            truncation=True,
            # TODO: what
            return_tensors='pt'
        )
        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']
        answer = torch.tensor(ord(self.data[index][2]) - ord('A'))
        # 一个Item代表一个文章、一个问题、一个答案
        # 说明文章的每个问题，拆分成了一个对应的Item
        # input_ids, attention_mask, answer = self.data[index]
        return input_ids, attention_mask, answer
