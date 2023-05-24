import os
import json

import torch
from transformers import RobertaTokenizer


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# 定义数据预处理函数
def preprocess_data(article, question, options, answer=None, max_length=512):
    inputs = tokenizer.encode_plus(question, article, add_special_tokens=True, max_length=max_length,
                                   padding='max_length', truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids'].squeeze()
    attention_mask = inputs['attention_mask'].squeeze()

    option_input_ids = []
    option_attention_masks = []
    for option in options:
        option_inputs = tokenizer.encode_plus(option, add_special_tokens=False, max_length=max_length,
                                              padding='max_length', truncation=True, return_tensors='pt')
        option_input_ids.append(option_inputs['input_ids'].squeeze())
        option_attention_masks.append(option_inputs['attention_mask'].squeeze())

    option_input_ids = torch.stack(option_input_ids)
    option_attention_masks = torch.stack(option_attention_masks)

    if answer is not None:
        # print(answer, type(answer))
        answer_label = ord(answer) - ord('A')  # 将答案转换为数字标签
        answer_one_hot = torch.zeros(len(options))
        answer_one_hot[answer_label] = 1
        return input_ids, attention_mask, option_input_ids, option_attention_masks, answer_one_hot

    return input_ids, attention_mask, option_input_ids, option_attention_masks

# 加载数据集
data_folder = '/home/cjx/python/roberta/RACE/dev/middle'  # 替换为实际的RACE数据集文件夹路径

processed_data = []
for file_name in os.listdir(data_folder):
    print(file_name)
    file_path = os.path.join(data_folder, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    article = data['article']
    questions = data['questions']
    options_list = data['options']
    answers_list = data['answers']

    for i, question in enumerate(questions):
        
        question_text = question.strip()
        options = options_list[i]
        answer = answers_list[i]

        # 截断较长的文本
        if len(article) > 512:
            article = article[:512]

        input_ids, attention_mask, option_input_ids, option_attention_masks, answer_one_hot = preprocess_data(article, question_text, options, answer)
        print(answer_one_hot)
        processed_data.append({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'option_input_ids': option_input_ids,
            'option_attention_masks': option_attention_masks,
            'answer': answer_one_hot
        })

print(type(processed_data))
# print(processed_data)


# if __name__ == "__main__":
    # main()