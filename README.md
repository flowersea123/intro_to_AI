# Introduction to AI：两个适合新手的机器学习项目

这个仓库用两个完整项目带你走过机器学习的基本流程：

1. **MNIST 手写数字识别**：这是一个十分类问题，使用全连接神经网络识别数字 0～9。
2. **加州房价预测**：这是一个回归问题，比较 KNN、线性回归、决策树和神经网络，并用交叉验证选择模型。

仓库同时保留原始 Jupyter Notebook、整理后的 Python 脚本、本地数据和中文学习笔记。推荐先看总学习路线，再运行项目，最后用复习清单自测。

## 你会学到什么

- 区分分类和回归任务
- 理解特征 `X`、标签 `y`、样本和模型
- 正确划分训练集、验证集和测试集
- 理解归一化、标准化、Pipeline 和数据泄漏
- 使用准确率、Log Loss、MSE、RMSE、R² 等指标
- 通过训练/测试差距识别过拟合
- 使用 K 折交叉验证和网格搜索选择超参数
- 从“代码能运行”进步到“知道为什么这样做”

## 仓库结构

```text
intro_to_AI/
├── notes/
│   ├── 00_学习路线与机器学习地图.md
│   ├── 01_MNIST手写数字识别笔记.md
│   ├── 02_加州房价预测笔记.md
│   └── 03_复习清单与自测题.md
├── projects/
│   ├── 01_mnist_recognition/
│   │   ├── MNIST_Data/
│   │   ├── MNIST_FNN_Student.ipynb
│   │   ├── mnist_fnn.py
│   │   └── README.md
│   └── 02_california_housing/
│       ├── data/california_housing.csv
│       ├── California Housing Price Reg - Student Version.ipynb
│       ├── california_housing.py
│       └── README.md
├── DATA_SOURCES.md
└── requirements.txt
```

## 环境准备

推荐 Python 3.11 或更新版本。在仓库根目录执行：

```bash
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 建议学习顺序

1. 阅读 [`notes/00_学习路线与机器学习地图.md`](notes/00_学习路线与机器学习地图.md)。
2. 阅读一个项目的笔记，不要求第一次全部记住。
3. 先用 `--quick` 跑通脚本，再运行完整版本。
4. 打开对应 notebook，逐单元修改参数并观察结果。
5. 不看笔记完成复习清单，最后再回来查漏补缺。

## 快速运行

MNIST 快速模式：

```bash
python projects/01_mnist_recognition/mnist_fnn.py --quick
```

加州房价快速模式：

```bash
python projects/02_california_housing/california_housing.py --quick
```

完整比较并对神经网络执行网格搜索：

```bash
python projects/02_california_housing/california_housing.py --tune
```

输出图表和结果表会写入各项目的 `artifacts/` 目录，该目录不会提交到 Git。

## 原始 notebook 的已有结果

这些结果来自本地 notebook 的一次运行。不同 Python、scikit-learn 版本和硬件可能产生小幅差异。

- MNIST：验证准确率 `0.9760`，测试准确率 `0.9759`，测试 Log Loss `0.0783`。
- 加州房价基础神经网络：测试 MSE `0.3126`，R² `0.7618`。
- 加州房价交叉验证后神经网络：测试 MSE `0.2664`，RMSE `0.5161`，R² `0.7971`。

## 数据和公开使用说明

数据来源、尺寸、校验值和引用信息见 [`DATA_SOURCES.md`](DATA_SOURCES.md)。

本仓库目前未指定代码许可证。公开仓库可供阅读和学习，但复制、修改或再发布前，应由仓库所有者补充明确许可证。
