# 代码使用

+ 按照`requirements.txt`配置环境
+ 将数据放置在目录下
+ 修改`train.py`中的路径为对应的数据路径

```python
    image_folder_path = "/home/liusr/plant_dataset/train/images"
    val_path = "/home/liusr/plant_dataset/val/images"
    test_path = "/home/liusr/plant_dataset/test/images"
    # CSV文件路径
    csv_file_path = "/home/liusr/plant_dataset/train/train_label.csv"
    val_csv = "/home/liusr/plant_dataset/val/val_label.csv"
    test_csv = "/home/liusr/plant_dataset/test/test_label.csv"
```

+ 运行指令`python train.py`

# 预训练模型

+ 下面代码中的resume设置为False则使用的是ImageNet上的预训练权重
+ 如果设置为True，则使用的是我保存下来的权重，为了更好地展示test_acc稳定提升的过程，我没有保存test_acc最好的权重，而是保存了Warmup之后的权重，模型收敛非常快，等待几轮epoch即可达到86%的test_acc
+ 模型权重文件

链接：https://pan.baidu.com/s/1_QTT2tkoxnWt1Vw8QpAz0w?pwd=1234 
提取码：1234 

```python
train(nb_epoch=1000,             # number of epoch
         batch_size=16,         # batch size
         store_name='bird',     # folder for output
         resume=True,          # resume training from checkpoint
         start_epoch=0,         # the start epoch number when you resume the training
         model_path='/home/liusr/PMG/bird/model.pth')         # the saved model where you want to resume the training

```

# 联系方式

若代码遇到任何问题，可以和我联系

+ VX:    15066883213
+ MAIL:    liushr6688@gmail.com
