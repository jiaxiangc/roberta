import os

import torch
import transformers
transformers.logging.set_verbosity_error()
from torch.utils.data import DataLoader

from models.RoBERTa import RoBERTaClassifier
from datasets.RACE_dataset import RACEDataset
from utils.misc import evaluate_model


def main():
    checkponits_path = 'checkpoints/roberta_race.pt'
    # 定义超参数
    device = torch.device('cuda:3')
    data_dir = './RACE/'
    batch_size = 4
    

    # 加载RACE数据集
    test_dataset = RACEDataset(os.path.join(data_dir, 'test'))
    print('Test dataset done')

    # 创建DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    print('Dataloader done')

    # 创建RoBERTa模型
    model = RoBERTaClassifier().to(device)
    model.load_state_dict(torch.load(checkponits_path))
    print('Mdeols done')

    accuracy = evaluate_model(model, test_loader, device)
    print(f'Accuracy on RACE test: {accuracy}')


if __name__ == "__main__":
    main()