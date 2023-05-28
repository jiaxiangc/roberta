import torch
from tqdm import tqdm






def evaluate_model(model, eval_loader, device):
    model.eval()
    # model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for input_ids, attention_mask, answer in tqdm(eval_loader,desc='Test RoBERTa'):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            answer = answer.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=answer)
            _, predicted = torch.max(outputs.logits, dim=1)
            total += answer.size(0)
            correct += (predicted == answer).sum().item()
    
    accuracy = correct / total
    return accuracy