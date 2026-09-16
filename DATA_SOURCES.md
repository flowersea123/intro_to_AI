# 数据来源与完整性说明

## 1. MNIST 手写数字数据

MNIST 是由手写数字灰度图组成的经典分类数据集。每张图片为 `28 × 28` 像素，像素原始取值为 0～255，标签为整数 0～9。

仓库保留原 notebook 实际读取的四个文件：

| 文件 | 形状 | 类型 | SHA-256 |
| --- | --- | --- | --- |
| `x_train.npy` | `(60000, 28, 28)` | `uint8` | `C5B45806D970E809A5F376F9B8461B7CA3B3E0A321282D72D06D00AB6C6C418F` |
| `y_train.npy` | `(60000,)` | `uint8` | `5DD4D822CAB3E20099239BC9D433D587AE3CE00E084D191079DD30B38380B336` |
| `x_test.npy` | `(10000, 28, 28)` | `uint8` | `4ACFA5C2911A2F95015EDA9A9B825FBD6BEC0F6A6F66942979B1473D33943A11` |
| `y_test.npy` | `(10000,)` | `uint8` | `FF7E84B144C037E7215DFA787D6773550C5DB83029D9A4E7BAE6E90F605F081D` |

本地原目录还存在 `mnist.npz`。经过逐数组比较，其中四个数组与上述 `.npy` 文件完全相同，因此没有重复收入仓库。

常用来源与背景：

- MNIST 数据集主页：<http://yann.lecun.com/exdb/mnist/>
- Keras 数据集说明：<https://keras.io/api/datasets/mnist/>

数据的原始作者、归属和许可不因本仓库公开而改变；再分发或商用前请核对原始来源的最新条款。

## 2. California Housing 数据

`california_housing.csv` 从本机已缓存的 `sklearn.datasets.fetch_california_housing(as_frame=True)` 数据导出，因此运行整理后的脚本不需要再次联网。

- 样本数：20,640
- 输入特征：8 个
- 目标列：`MedHouseVal`
- CSV 总列数：9 个
- 目标含义：街区房屋价值中位数，以 10 万美元为单位
- CSV SHA-256：`94EF369B909F5C725340B512B7C9A05C1550A554B6592D29303990E3BA521A1F`

特征包括收入中位数、房龄、平均房间数、平均卧室数、人口、平均居住人数、纬度和经度。该数据来自 1990 年美国人口普查，适合教学，但不代表今天的市场，也不能直接用于真实估值决策。

参考：

- scikit-learn 数据集说明：<https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html>
- Pace, R. Kelley and Ronald Barry, *Sparse Spatial Autoregressions*, 1997.

## 3. 为什么没有上传课程 PDF

PDF 不是代码运行所必需的数据，并且本地文件没有附带可核验的公开再分发许可。为避免把第三方课程材料误当作本仓库原创内容，本仓库只整理项目代码、运行数据和原创学习笔记。
