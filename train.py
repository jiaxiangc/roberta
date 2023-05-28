import os

import torch
import torch.nn as nn
import torch.distributed as dist
import transformers
transformers.logging.set_verbosity_error()
from transformers import RobertaForMultipleChoice, TrainingArguments, Trainer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from datasets.RACE_dataset import RACEDataset
from models.RoBERTa import RoBERTaClassifier
from utils.misc import evaluate_model

local_rank = int(os.environ['LOCAL_RANK'])

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
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    dist.barrier()
    setup_for_distributed(local_rank == 0)



def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    # model.to(device)
    # TODO: where is epochs
    for input_ids, attention_mask, answer in tqdm(train_loader, desc=f'Train'):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        answer = answer.to(device)
        # print("answer: ", answer.shape)
        # 标准化流程
        # print(f'input_ids {input_ids.shape}')
        # print(f'attention_mask {attention_mask.shape}')
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=answer)
        # print(f'Logits shape: {logits.shape}')
        # loss = criterion(logits, answer)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('Loss: {:.3f}'.format(loss.item()))

def main():
    dist_init()

    # 定义超参数
    device = torch.device('cuda')
    data_dir = './RACE/'
    batch_size =12
    num_epochs = 4
    learning_rate = 1e-5
    output_dir = './checkpoints/'
    checkponits_name = 'roberta_race'

    # 加载RACE数据集
    train_dataset = RACEDataset(os.path.join(data_dir, 'train'))
    train_sampler = DistributedSampler(train_dataset)
    print('Train dataset done')
    dev_dataset = RACEDataset(os.path.join(data_dir, 'dev'))
    dev_sampler = DistributedSampler(dev_dataset)
    print('Dev dataset done')

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, sampler=dev_sampler)
    print('Dataloader done')

    # 创建RoBERTa模型
    # model = RoBERTaClassifier().to(device)
    model = RobertaForMultipleChoice.from_pretrained('roberta-base').to(device)
    model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    print('Mdeols done')

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)


    # Trainer




    print('====== Staring training ======')
    # 训练和评估模型
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_model(model, train_loader, criterion, optimizer, device)
        # 单机多卡保存权重
        if dist.get_rank() == 0:
            torch.save(model.state_dict(), f'{os.path.join(output_dir, checkponits_name)}_{epoch}.pt')
            print(f'Writing to {os.path.join(output_dir, checkponits_name)}')
            accuracy = evaluate_model(model, dev_loader, device)
            print(f'Epoch {epoch+1}/{num_epochs}: Accuracy: {accuracy}')

    print('====== Training finshed ======')


if __name__ == "__main__":
    main()