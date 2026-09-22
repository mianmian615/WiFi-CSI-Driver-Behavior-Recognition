# WiFi-CSI-Driver-Behavior-Recognition
基于 WiFi CSI 信号的危险驾驶行为识别研究项目，由学生团队与指导老师共同完成，主要研究 WiFi CSI 信号处理、跨域学习及深度学习方法在驾驶行为识别中的应用。
# WiFi CSI Driver Behavior Recognition

基于 WiFi CSI（Channel State Information，信道状态信息）的驾驶行为识别研究项目。

本项目由**学生团队与指导老师共同完成**，主要研究 WiFi CSI 信号处理、深度学习以及跨域学习方法在驾驶行为识别中的应用，探索利用无线信号实现非接触式驾驶行为感知与识别。

## 项目简介

驾驶过程中的危险行为会对交通安全产生重要影响。传统驾驶行为监测方法通常依赖摄像头、可穿戴设备或其他传感器，而 WiFi CSI 能够反映无线信号传播过程中由人体运动引起的信道变化，因此可以作为一种非接触式行为感知手段。

本项目以 WiFi CSI 信号为主要研究对象，通过信号处理、特征学习和深度学习模型，对驾驶过程中的行为进行识别，并进一步探索不同环境、不同数据域之间的泛化能力。

## 主要研究内容

* WiFi CSI 信号数据处理
* CSI 信号特征提取与预处理
* 驾驶行为识别
* 深度学习模型构建
* 跨域学习（Domain Generalization）
* 不同数据域之间的模型泛化
* WiFi 感知相关算法研究

## 项目结构

```text
WiFi-CSI-Driver-Behavior-Recognition/
│
├── CSI/
│   ├── csi_dg.py
│   ├── csi_domainset.py
│   └── data_process/
│       ├── extract_csi.py
│       └── signal_process.py
│
├── algorithms/
│   ├── algorithms.py
│   ├── crossgrad.py
│   └── mmd_aae.py
│
├── models/
│   ├── InceptionTime.py
│   ├── ResNet1D_group.py
│   ├── ResNet2Dnew.py
│   ├── backbones.py
│   ├── networks.py
│   ├── resnet1d.py
│   ├── resnet1d_rfid.py
│   ├── resnet2d.py
│   └── ...
│
├── lib/
│   ├── fast_data_loader.py
│   └── misc.py
│
├── datasets.py
├── hparams_registry.py
├── requirements.txt
└── README.md
```

## 环境配置

建议使用 Python 3.x 环境运行本项目。

首先创建并激活虚拟环境：

```bash
python -m venv venv
```

Windows：

```bash
venv\Scripts\activate
```

Linux / macOS：

```bash
source venv/bin/activate
```

然后安装项目依赖：

```bash
pip install -r requirements.txt
```

## 数据说明

本项目涉及 WiFi CSI 数据处理。

由于原始 CSI 数据可能包含实验参与者信息、实验环境信息以及其他需要保护的数据，因此**原始数据不随本代码仓库公开上传**。

运行项目时，请根据实际实验数据情况配置本地数据路径。

建议的数据组织方式根据具体数据集和实验设置进行调整。

## 数据处理流程

项目整体的数据处理流程可以概括为：

```text
WiFi 信号
    ↓
CSI 数据获取
    ↓
CSI 信号提取
    ↓
信号预处理
    ↓
特征构建
    ↓
深度学习模型
    ↓
驾驶行为识别
```

## 模型与算法

项目中包含多种深度学习模型和跨域学习算法，用于探索 WiFi CSI 驾驶行为识别任务。

主要包括：

* ResNet 系列模型
* InceptionTime
* CNN / 深度神经网络相关模型
* CrossGrad
* MMD-AAE
* Domain Generalization 相关方法

不同模型和算法用于比较不同特征学习方法及跨域学习方法在 WiFi CSI 行为识别任务中的表现。

## 运行说明

具体运行方式需要根据实际使用的数据集、数据路径以及实验配置进行调整。

在运行前，请确认：

1. 已正确安装 `requirements.txt` 中的依赖；
2. 已准备对应的 CSI 数据；
3. 已根据本地环境配置数据路径；
4. 已根据实验需求设置模型和超参数。

## 项目状态

本项目目前主要用于科研与课程项目研究，代码会根据实验过程持续整理和完善。

后续可能进一步补充：

* 更完整的数据处理说明
* 实验配置
* 训练方法
* 实验结果
* 模型对比
* 可视化结果
* 更详细的使用示例

## 研究团队

本项目由**学生团队与指导老师共同完成**，用于 WiFi CSI 驾驶行为识别相关研究。

## 注意事项

本仓库主要公开项目代码。

原始实验数据、涉及个人信息的数据以及其他未经授权公开的数据不包含在本仓库中。

如果使用本项目中的代码或研究思路，请根据实际情况进行引用，并遵守相关数据集及研究项目的使用规定。

## License

本项目的具体开源许可协议将在后续根据项目要求确定。
