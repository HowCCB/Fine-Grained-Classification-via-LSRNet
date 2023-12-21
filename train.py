from __future__ import print_function
import os
from PIL import Image

import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils import *
import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler
from PIL import  Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
import pandas as pd
import os
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from torch.utils.tensorboard import SummaryWriter
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def count_identical_rows(tensor1, tensor2):
    
    # 比较两个张量的对应元素是否相等
    equal_elements = torch.eq(tensor1, tensor2)
    # 沿着行维度进行逻辑与操作，得到每行是否完全相同的布尔值张量
    rows_equal = torch.all(equal_elements, dim=1)
    # 统计每行相同的数量
    num_same_rows = torch.sum(rows_equal).item()
    # print("有", num_same_rows, "行是完全相同的。")
    return num_same_rows
class CustomDataset(Dataset):
    def __init__(self, image_folder_path, csv_file_path, transform=None):
        self.image_folder_path = image_folder_path
        self.csv_file_path = csv_file_path
        self.transform = transform
        self.data = pd.read_csv(self.csv_file_path)
        self.label_encoder = MultiLabelBinarizer()
        self.num_classes = len(self.data['labels'].unique()) # 获取类别数量

        # 对标签进行编码
        labels = [labels_str.split(' ') for labels_str in self.data['labels']]
        self.label_encoder.fit(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.image_folder_path, img_name)
        image = Image.open(img_path).convert("RGB")
        labels = self.data.iloc[idx, 1]
        labels_list = labels.split(' ')
        labels_encoded = self.label_encoder.transform([labels_list])
        label_one_hot = torch.from_numpy(labels_encoded.sum(axis=0)).float()

        if self.transform is not None:
            image = self.transform(image)
        return image, label_one_hot


    def get_label(self,mul_hot_code):
    # labels = []
        indices = np.where(mul_hot_code == 1)[0] # 获取所有值为1的索引
    # for index in indices:
    # label = self.label_encoder.inverse_transform(np.array([[index]]))[0] # 获取对应的原始标签
    # labels.append(label)
    # return labels
        label = self.label_encoder.inverse_transform(np.array([[indices]]))
        return label


def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # Data
    print('==> Preparing data..')
    data_transform = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # trainset = torchvision.datasets.ImageFolder(root='./bird/train', transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
       # Data loading code
    # data_transform = transforms.Compose([
    #     transforms.Resize((128, 128)), # 调整图像大小
    #     transforms.ToTensor(), # 转换为Tensor
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # 标准化
    #     ])
    image_folder_path = "/home/liusr/plant_dataset/train/images"
    val_path = "/home/liusr/plant_dataset/val/images"
    test_path = "/home/liusr/plant_dataset/test/images"
    # CSV文件路径
    csv_file_path = "/home/liusr/plant_dataset/train/train_label.csv"
    val_csv = "/home/liusr/plant_dataset/val/val_label.csv"
    test_csv = "/home/liusr/plant_dataset/test/test_label.csv"
    train_dataset = CustomDataset(image_folder_path, csv_file_path, transform=data_transform)
    # print(train_dataset[0].shape)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    writer = SummaryWriter('logs')
    # Model
    if resume:
        net = torch.load(model_path)
    else:
        net = load_model(model_name='resnet50_pmg', pretrain=True, require_grad=True)
    netp = torch.nn.DataParallel(net, device_ids=[0])

    # GPU
    device = torch.device("cuda:0")
    net.to(device)
    # cudnn.benchmark = True

    CELoss = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.SGD([
        {'params': net.classifier_concat.parameters(), 'lr': 0.002},
        {'params': net.conv_block1.parameters(), 'lr': 0.002},
        {'params': net.classifier1.parameters(), 'lr': 0.002},
        {'params': net.conv_block2.parameters(), 'lr': 0.002},
        {'params': net.classifier2.parameters(), 'lr': 0.002},
        {'params': net.conv_block3.parameters(), 'lr': 0.002},
        {'params': net.classifier3.parameters(), 'lr': 0.002},
        {'params': net.features.parameters(), 'lr': 0.0002}

    ],
        momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002]
    rem = []
    los = []
    test_rem = []
    test_los = []
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            #每个batch传入16个数据
            idx = batch_idx   #这个idx就是1/2/3/4/5
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # update learning rate
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])


            temp =  random.uniform(0,1)
            a = 1-sigmoid(epoch)
            # Step 1
            # if temp>a:
            optimizer.zero_grad()   
            inputs1 = jigsaw_generator(inputs, 8)
            output_1, _, _, _ = netp(inputs1)
            loss1 = CELoss(output_1, targets) * 1
            loss1.backward()
            optimizer.step()

        # b = 1-sigmoid(epoch)
        # if temp>a:
            # Step 2
            optimizer.zero_grad()
            inputs2 = jigsaw_generator(inputs, 4)
            _, output_2, _, _ = netp(inputs2)
            loss2 = CELoss(output_2, targets) * 1
            loss2.backward()
            optimizer.step()



            # c = sigmoid(epoch)
            # if temp>c:
                # Step 3
            optimizer.zero_grad()
            inputs3 = jigsaw_generator(inputs, 2)
            _, _, output_3, _ = netp(inputs3)
            loss3 = CELoss(output_3, targets) * 1
            loss3.backward()
            optimizer.step()
            # d = sigmoid(epoch)
            # if temp>d:
            # Step 4
            optimizer.zero_grad()
            _, _, _, output_concat = netp(inputs)
            concat_loss = CELoss(output_concat, targets) * 2
            concat_loss.backward()
            optimizer.step()
            #  training log
            # _, predicted = torch.max(output_concat.data, dim = 1)
            total += targets.size(0)
            # print(total)
            # print(output_concat)
            predicted =(output_concat>0.5).float()
            # print(predicted-targets.data)
            # print(predicted)
            # correct += predicted.eq(targets.data).cpu().sum()
            correct  = correct+count_identical_rows(predicted,targets.data)
            train_loss += (loss1.item() + loss2.item() + loss3.item() + concat_loss.item())
            train_loss1 += loss1.item()
            train_loss2 += loss2.item()
            train_loss3 += loss3.item()
            train_loss4 += concat_loss.item()

            if batch_idx % 50 == 0:
                print(
                    'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                    train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total))
        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        rem.append(train_acc)
        los.append(train_loss)
        print("train_acc",train_acc)
        writer.add_scalar('Train/Loss', train_loss, global_step=epoch)
        for name, param in netp.named_parameters():
             writer.add_histogram(name, param, global_step=epoch)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
                epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1), train_loss3 / (idx + 1),
                train_loss4 / (idx + 1)))

        # if epoch < 5 or epoch >= 80:
        if True:
            val_acc, val_acc_com, val_loss = test(net, CELoss, 3)
            test_rem.append(val_acc)
            test_los.append(val_loss)
            if val_acc_com > max_val_acc:
                max_val_acc = val_acc_com
                net.cpu()
                torch.save(net, './' + store_name + '/model.pth')
                net.to(device)

            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('Iteration %d, test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                epoch, val_acc, val_acc_com, val_loss))

        
        else:
            net.cpu()
            torch.save(net, './' + store_name + '/model.pth')
            net.to(device)
        print("rem",rem)
        print("los",los)
        print("test_rem",test_rem)
        print("test_los",test_los)

train(nb_epoch=1000,             # number of epoch
         batch_size=16,         # batch size
         store_name='bird',     # folder for output
         resume=True,          # resume training from checkpoint
         start_epoch=0,         # the start epoch number when you resume the training
         model_path='/home/liusr/PMG/bird/model.pth')         # the saved model where you want to resume the training
