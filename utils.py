import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from model import *
from Resnet import *
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


def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)


def load_model(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PMG(net, 512, 6)    #这个地方有改动，将200改成6了

    return net


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))


def jigsaw_generator(images, n):
    l = []
    for a in range(n):
        for b in range(n):
            l.append([a, b])
    block_size = 448 // n
    rounds = n ** 2
    random.shuffle(l)
    jigsaws = images.clone()
    for i in range(rounds):
        x, y = l[i]
        temp = jigsaws[..., 0:block_size, 0:block_size].clone()
        jigsaws[..., 0:block_size, 0:block_size] = jigsaws[..., x * block_size:(x + 1) * block_size,
                                                y * block_size:(y + 1) * block_size].clone()
        jigsaws[..., x * block_size:(x + 1) * block_size, y * block_size:(y + 1) * block_size] = temp

    return jigsaws


def test(net, criterion, batch_size):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    correct_com = 0
    total = 0
    idx = 0
    device = torch.device("cuda:0")

    # transform_test = transforms.Compose([
    #     transforms.Resize((550, 550)),
    #     transforms.CenterCrop(448),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # # ])
    # testset = torchvision.datasets.ImageFolder(root='./bird/test',
    #                                            transform=transform_test)
    val_path = "/home/liusr/plant_dataset/test/images"
    test_path = "/home/liusr/plant_dataset/test/images"
    # CSV文件路径
    csv_file_path = "/home/liusr/plant_dataset/train/train_label.csv"
    val_csv = "/home/liusr/plant_dataset/test/test_label.csv"
    test_csv = "/home/liusr/plant_dataset/test/test_label.csv"
    data_transform = transforms.Compose([
    transforms.Resize((550, 550)),
    transforms.RandomCrop(448, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
    testset = CustomDataset(test_path, test_csv, transform=data_transform)
    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(testloader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)   #这部分没有必要在意
        output_1, output_2, output_3, output_concat= net(inputs)
        outputs_com = output_1 + output_2 + output_3 + output_concat

        loss = criterion(output_concat, targets)

        test_loss += loss.item()
        # _, predicted = torch.max(output_concat.data, 1)
        # _, predicted_com = torch.max(outputs_com.data, 1)
        total += targets.size(0)
        # correct += predicted.eq(targets.data).cpu().sum()
        # correct_com += predicted_com.eq(targets.data).cpu().sum()
        def count_identical_rows(tensor1, tensor2):
            # 比较两个张量的对应元素是否相等
            equal_elements = torch.eq(tensor1, tensor2)
            # 沿着行维度进行逻辑与操作，得到每行是否完全相同的布尔值张量
            rows_equal = torch.all(equal_elements, dim=1)
            # 统计每行相同的数量
            num_same_rows = torch.sum(rows_equal).item()
            # print("有", num_same_rows, "行是完全相同的。")
            return num_same_rows
        predicted =(output_concat>0.5).float()
        correct +=count_identical_rows(predicted,targets.data)
        predicted_com = (outputs_com>0.5).float()
        correct_com +=count_identical_rows(predicted_com,targets.data)
        if batch_idx % 50 == 0:
            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d) |Combined Acc: %.3f%% (%d/%d)' % (
            batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total, 100. * float(correct_com) / total, correct_com, total))

    test_acc = 100. * float(correct) / total
    test_acc_en = 100. * float(correct_com) / total
    test_loss = test_loss / (idx + 1)

    return test_acc, test_acc_en, test_loss


