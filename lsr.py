# # # import torch

# # # # 两个示例张量
# # # tensor1 = torch.tensor([[1, 2, 3],
# # #                         [4, 5, 6],
# # #                         [7, 8, 9]])

# # # tensor2 = torch.tensor([[1, 2, 3],
# # #                         [4, 5, 6],
# # #                         [7, 8, 3]])

# # # # 比较两个张量的对应元素是否相等
# # # equal_elements = torch.eq(tensor1, tensor2)
# # # print(equal_elements)
# # # # 沿着行维度进行逻辑与操作，得到每行是否完全相同的布尔值张量
# # # rows_equal = torch.all(equal_elements, dim=1)
# # # print(rows_equal)
# # # # 统计每行相同的数量
# # # num_same_rows = torch.sum(rows_equal).item()
# # # print(torch.sum(rows_equal))
# # # print("有", num_same_rows, "行是完全相同的。")
# from Evison import Display, show_network
# from torchvision import models
# from PIL import Image
# import torch
# import torchvision.transforms as transforms
# # 使用show_network这个辅助函数来看看有什么网络层(layers)
# # show_network(network)
# # network = models.efficientnet_b0(pretrained=True)
# network = torch.load('/home/liusr/bird/model.pth')
# network.cpu()
# show_network(network)
# visualized_layer = 'features'
# display = Display(network, visualized_layer, img_size=(550, 550))  # img_size的参数指的是输入图片的大小
# image = Image.open('/home/liusr/PMG/8a2d598f2ec436e6.jpg').resize((550, 550))
# display.save(image)







# # network = torch.load('/home/liusr/PMG/bird/model.pth')
# # device = torch.device("cuda:0")
# # network.to(device)
# # network.eval()
# # data_transform = transforms.Compose([
# #     transforms.Resize((550, 550)),
# #     transforms.RandomCrop(448, padding=8),
# #     transforms.RandomHorizontalFlip(),
# #     transforms.ToTensor(),
# #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# # ])
# # image = Image.open('/home/liusr/PMG/8a2d598f2ec436e6.jpg').convert("RGB")
# # image_tensor = data_transform(image).unsqueeze(0)  # 
# # image_tensor = image_tensor.to(device)
# # visualized_layer = 'conv_block3.1.bn'
# # display = Display(network, visualized_layer, img_size=(550, 550))  # img_size的参数指的是输入图片的大小
# # # output = network(image_tensor)
# # # print(output)
# # display.save(image)
from Evison import Display, show_network
from torchvision import models
from PIL import Image
network = models.efficientnet_b0(pretrained=True)
image = Image.open('/home/liusr/PMG/8a2d598f2ec436e6.jpg').resize((550, 550))
visualized_layer = 'features.7.0'
display = Display(network, visualized_layer, img_size=(550, 550))  
display.save(image)