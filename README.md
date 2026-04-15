# MNIST-C

一个尽量精简的纯 C99 MNIST 手写数字识别项目。

## 特性

- 纯 C 实现，不依赖任何第三方库
- 三层全连接网络：`784 -> 128 -> 10`
- 隐藏层 `ReLU`，输出层 `Softmax`，损失函数为交叉熵
- 小批量 SGD 训练
- He 初始化，正态分布使用 Box-Muller 生成
- `make data` 自动下载并校验 MNIST 数据集
- 训练完成后自动将参数导出到头文件 `src/model_params.h`
- `make verify` 随机抽取测试集样本，显示字符画和预测结果

## 目录结构

- `src/`：源代码与模型参数头文件
- `build/`：编译产生的 `.o` 文件
- `bin/`：可执行文件
- `data/`：MNIST 数据集
- `scripts/`：数据下载脚本

## 使用方法

### 1. 下载数据集

```bash
make data
```

如果默认下载源不可用，可以手动指定镜像前缀：

```bash
MNIST_BASE_URL='https://your-mirror.example/mnist' make data
```

### 2. 编译项目

```bash
make
```

### 3. 开始训练

默认训练 3 个 epoch：

```bash
make train
```

也可以自定义参数：

```bash
make train ARGS='--epochs 5 --batch-size 64 --learning-rate 0.01 --seed 42'
```

训练结束后会生成 `src/model_params.h`，随后 `verify` 会自动使用这份参数。

### 4. 随机验证

```bash
make verify
```

程序会：

- 从测试集中随机挑选一张图片
- 在终端打印字符画
- 输出真实标签和预测标签
- 显示 `CORRECT` 或 `WRONG`

## 可执行文件

- `bin/train`：训练模型并导出参数头文件
- `bin/verify`：随机验证一张测试图片

## 清理

```bash
make clean
```

## 说明

- 首次运行前请先执行 `make data`
- `verify` 依赖 `src/model_params.h` 中的已训练参数，因此需要先执行 `make train`
- 默认目标是“项目尽量精简且好用”，因此实现上保持为最小可维护版本
