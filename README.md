电力设备类-受电弓设备检测数据集 深度学习YOLOv11 受电弓检测系统![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/df40cfabb67f464285d848adec7ea56e.png)
，2676张，yolo，voc，coco三种标注方式
图像尺寸:640*640
类别数量:3类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e61aa5c214084924ac2cf2cc333d3151.png)

训练集图像数量:2275; 验证集图像数量:265； 测试集图像数量:136
类别名称: 每一类图像数 ，每一类标注数![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0e9e6a66971a4e579d8c79fbe825bf10.png)

条带 / 钢带 2675,2677
接触点 2625,2649
立柱 / 桅杆 786,795
image num: 267
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/960cb32980f348afa6670c215d621a3a.png)


模型代码：提供全部训练及测试源代码，模型训练使用yolov11n训练，50个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e97a71c7d4d1457e85edb186ec1cc3dd.png)

qt界面：运行界面采用pyqt5编写， 提供全部源代码，支持图片、视频及摄像头进行检测: 界面可实时显示目标位置、目标总数、置信度等信息: 支持检测结果保存;本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/07ba470b087e4e2eabec2b9b253e7df8.png)
# YOLOv11 受电弓检测简易训练代码
> 类别：`strip`(受电弓滑板)、`mast`(支架)、`contact_point`(接触点)

## 1. dataset.yaml 配置文件
```yaml
path: ./dataset  # 数据集根目录
train: images/train
val: images/val
test: images/test

names:
  0: strip
  1: mast
  2: contact_point
```

## 2. train.py 训练脚本
```python
from ultralytics import YOLO

# 加载模型，yolov11n 轻量版
model = YOLO("yolov11n.pt")

if __name__ == "__main__":
    results = model.train(
        data="dataset.yaml",
        epochs=80,
        imgsz=640,
        batch=8,
        device=0,  # GPU训练，无GPU改为 device="cpu"
        project="train_result",
        name="pantograph",
        save=True,
        val=True
    )
    print("训练完成！权重保存在 train_result/pantograph/weights/best.pt")
```

## 3. 推理预测简易代码（测试图片）
```python
from ultralytics import YOLO

model = YOLO("train_result/pantograph/weights/best.pt")

# 单张图片推理
res = model.predict(source="test.jpg", save=True, conf=0.5)
for r in res:
    print(r.boxes.cls, r.boxes.conf)
```

## 4. 环境安装
```bash
pip install ultralytics opencv-python
```

## 项目说明
- 任务：受电弓部件目标检测，3个类别
- 训练输出：best.pt权重、PR曲线、loss曲线，和你截图里的mAP结果对应
- 训练完可直接接入PyQt5界面做可视化检测

