# 项目一：MNIST 手写数字识别

目标：输入一张 `28 × 28` 的灰度手写数字图片，输出它最可能属于 0～9 中的哪一类。

## 文件

- `MNIST_FNN_Student.ipynb`：原始 notebook，保留训练输出和可视化。
- `mnist_fnn.py`：整理后的可运行脚本。
- `MNIST_Data/`：训练集和测试集的图片、标签。

## 运行

从仓库根目录执行：

```bash
python projects/01_mnist_recognition/mnist_fnn.py --quick
```

快速模式会抽取较小的数据子集并减少训练轮数，适合确认环境。完整训练：

```bash
python projects/01_mnist_recognition/mnist_fnn.py
```

脚本会输出验证集与测试集指标，并在 `artifacts/mnist_loss.png` 保存损失曲线。

学习前后请配合阅读 [`../../notes/01_MNIST手写数字识别笔记.md`](../../notes/01_MNIST手写数字识别笔记.md)。
