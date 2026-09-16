# 项目二：加州房价预测

目标：根据 8 个街区层面的特征预测房屋价值中位数，并比较不同回归模型。

## 文件

- `California Housing Price Reg - Student Version.ipynb`：原始 notebook，包含四模型比较和网格搜索结果。
- `california_housing.py`：整理后的可运行脚本。
- `data/california_housing.csv`：本地数据，使脚本可以离线运行。

## 运行

快速检查：

```bash
python projects/02_california_housing/california_housing.py --quick
```

完整四模型比较：

```bash
python projects/02_california_housing/california_housing.py
```

完整比较并执行 5 折网格搜索：

```bash
python projects/02_california_housing/california_housing.py --tune
```

`--tune` 会训练 9 组超参数、每组做 5 折验证，耗时明显更长。结果表和图表保存到 `artifacts/`。

学习前后请配合阅读 [`../../notes/02_加州房价预测笔记.md`](../../notes/02_加州房价预测笔记.md)。
