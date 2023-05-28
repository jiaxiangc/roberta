import json
import re
import os

import torch
import torch.distributed as dist
import transformers
transformers.logging.set_verbosity_error()

from torch.nn.parallel import DistributedDataParallel


from transformers import RobertaTokenizer, RobertaForMultipleChoice

from models.RoBERTa import RoBERTaClassifier

# local_rank = int(os.environ['LOCAL_RANK'])

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def dist_init():
    # torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    # dist.barrier()
    # setup_for_distributed(local_rank == 0)


def main():
    # dist_init()
    dic = {0: "A", 1: "B", 2: "C", 3: "D"}
    # 从一个文件中加载：1 article，1 question，4 answer
    device = torch.device('cuda')
    checkpinits = 'checkpoints/roberta_race.pt'
 
    res = []
    file_path = './RACE/dev/middle/13.txt'

    print('Loading RoBERTa Tokenizer')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaForMultipleChoice.from_pretrained('roberta-base').to(device)
    # model = DistributedDataParallel(model)
    # state_dict = torch.load(checkpinits)
    # print(state_dict.items())

    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(checkpinits).items()})
    # quit()

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        data = json.loads(content)

        article = data['article'].replace('\n', ' ')
        article = re.sub(r"\s+", " ", article)

        questions = data['questions']
        options = data['options']
        answers = data['answers']
        print("article: \n", article)
        print("answers: \n", answers)
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

            # print(qa_list)
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
        answer = torch.tensor(ord(d[2]) - ord('A')).unsqueeze(0).to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=answer)
        print('logits: ', torch.softmax(outputs.logits, dim=1))
        print('predict answer: ', dic[torch.max(outputs.logits, dim=1).indices.item()])
    # print('predict answer: ', answers)
        



if __name__ == '__main__':
    main()