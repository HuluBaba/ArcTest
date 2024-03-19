# script to perform classification on ImageNet100 ultilizing ResNet50

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import datasets as mydatasets
from torch.utils.data import DataLoader

from tqdm import tqdm
import yaml




if __name__=='__main__':
    # load configuration in yaml
    config_path="./config.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = config['device']
    if device == 'xpu':
        import intel_extension_for_pytorch as ipex
    dataset_path = config['dataset_path']
    num_class = config['num_class']
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['lr']
    lr_decay = config['lr_decay']
    weight_decay = config['weight_decay']
    optimizer_type = config['optimizer_type']
    



    # Data load
    train_dataset, test_dataset, label_names = mydatasets.load_imagenet100_data(dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Make Model
    model = torchvision.models.resnet50()
    model.load_state_dict(torch.load("D:/Code/CVModel/pretrained_weights/resnet50-0676ba61.pth"))

    model.fc = nn.Linear(2048, num_class)
    for param in model.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True



    # Fine-tune
    loss_fn = nn.CrossEntropyLoss()

    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    if device == 'cuda':
        model.to('cuda')
    elif device == 'xpu':
        model.to('xpu')
        loss_fn.to('xpu')
        model, optimizer = ipex.optimize(model, optimizer=optimizer)

    if lr_decay == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss = 10000.0
    with tqdm(total=epochs*len(train_loader),postfix=f"{loss:7.2f}",leave=True) as abar:
        for epoch in range(1,epochs+1):
            for idx,(img,label) in enumerate(train_loader):
                img = img.to(device)
                pred = model(img)
                label = label.to(device)
                label_onehot = F.one_hot(label, num_class).float()
                loss = loss_fn(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                abar.update(1)
                abar.set_postfix_str(f"{loss:7.2f}")
                if (idx+1)%100==0:
                    model.eval()
                    right_num = 0
                    total_num = 0
                    with torch.no_grad():
                        for imgs, labels in test_loader:
                            imgs = imgs.to(device)
                            labels = labels.to(device)
                            pred = model(imgs)
                            pred_label = torch.argmax(pred, dim=1)
                            right_num += torch.sum(pred_label==labels)
                            total_num += len(labels)
                    print(f"right_num:{right_num}, total_num:{total_num}, acc:{right_num/total_num}")
                    model.train()
            scheduler.step()
            # Valid
            if epoch%config['valid_per_epoch'] == 0:
                model.eval()
                right_num = 0
                total_num = 0
                with torch.no_grad():
                    for imgs, labels in test_loader:
                        imgs = imgs.to(device)
                        labels = labels.to(device)
                        pred = model(imgs)
                        pred_label = torch.argmax(pred, dim=1)
                        right_num += torch.sum(pred_label==labels)
                        total_num += len(labels)
                print(f"right_num:{right_num}, total_num:{total_num}, acc:{right_num/total_num}")
                model.train()
            # save pth
            if epoch%config['save_per_epoch'] == 0:
                torch.save(model.state_dict(), f"./model_epoch{epoch}.pth")  
                torch.save(optimizer.state_dict(), f"./optimizer_epoch{epoch}.pth")

            
            





    right_num = 0
    total_num = 0
    torch.no_grad()
    with tqdm(total=len(test_loader)) as pbar:
        for img,label in test_loader:
            img = img.to('cuda')
            label = label.to('cuda')
            pred = model(img)
            pred_label = torch.argmax(pred, dim=1)
            right_num += torch.sum(pred_label==label)
            total_num += len(label)
            pbar.update(1)

    print(f"right_num:{right_num}, total_num:{total_num}, acc:{right_num/total_num}")