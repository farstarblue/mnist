# MNIST-C

一个尽量精简的纯 C99 MNIST 手写数字识别项目。

## 特性

- 纯 C 实现，不依赖任何第三方库
- 三层全连接网络：`784 -> 128 -> 10`
- 隐藏层 `ReLU`，输出层 `Softmax`，损失函数为交叉熵
- 小批量 SGD 训练
- He 初始化，正态分布使用 Box-Muller 生成
- 训练参数统一放在头文件 `src/config.h`
- `make data` 自动下载并校验 MNIST 数据集
- 训练完成后自动将参数导出到头文件 `src/model_params.h`
- `make verify` 随机抽取测试集样本，显示字符画和预测结果
- 训练时按 batch 实时输出进度
- 提供独立的 RISC-V 交叉编译入口 `Makefile.riscv`

## 目录结构

- `src/`：源代码与模型参数头文件
- `build/`：编译产生的 `.o` 文件
- `bin/`：可执行文件
- `build-riscv/`：RISC-V 交叉编译产生的 `.o` 文件
- `bin-riscv/`：RISC-V 交叉编译产生的可执行文件
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

训练参数位于 `src/config.h`，包括：

- `TRAIN_EPOCHS`
- `TRAIN_BATCH_SIZE`
- `TRAIN_LEARNING_RATE`
- `TRAIN_RANDOM_SEED`

修改这些宏后，直接重新执行训练即可。默认训练 3 个 epoch：

```bash
make train
```

训练结束后会生成 `src/model_params.h`，随后 `verify` 会自动使用这份参数。
训练过程中会在终端按 batch 刷新当前 epoch 的进度和 batch loss。

### 4. 随机验证

```bash
make verify
```

程序会：

- 从测试集中随机挑选一张图片
- 在终端打印字符画
- 输出真实标签和预测标签
- 显示 `CORRECT` 或 `WRONG`

### 5. RISC-V 交叉编译

Linux 本机直接使用默认 `Makefile` 训练和验证。
如果需要生成 RISC-V 版本，可使用独立的 `Makefile.riscv`：

```bash
make -f Makefile.riscv
```

也可以通过主 `Makefile` 的快捷目标调用：

```bash
make riscv
make riscv-train
make riscv-verify
make riscv-train-scalar
make riscv-train-vector
make riscv-verify-scalar
make riscv-verify-vector
```

默认会同时生成 4 个 RISC-V ELF：

- `bin-riscv/train-scalar.elf`
- `bin-riscv/verify-scalar.elf`
- `bin-riscv/train-vector.elf`
- `bin-riscv/verify-vector.elf`

其中：

- scalar 版本使用 `-march=rv64gc_zicsr -mabi=lp64d`
- vector 版本使用 `-march=rv64gcv_zicsr -mabi=lp64d`
- vector 版本在 `src/network.c` 的热点计算路径中显式使用 RVV intrinsics，因此会生成真实的 RVV 指令，而不是仅依赖编译器自动向量化

如需覆盖架构或 ABI，可传入：

```bash
make -f Makefile.riscv \
  SCALAR_ARCH=rv64gc_zicsr \
  VECTOR_ARCH=rv64gcv_zicsr \
  ABI=lp64d
```

> 注意：RISC-V 交叉编译只生成目标架构可执行文件，不会在当前 x86 Linux 主机上直接运行。
如果 `riscv64-elf-gcc` 与标准库/sysroot 是分开安装的，可以额外指定：

```bash
make -f Makefile.riscv SYSROOT=/path/to/riscv-sysroot
```

`Makefile.riscv` 不会使用 `riscv64-linux-gnu-gcc`；如果当前 bare-metal 工具链缺少标准库或 `sysroot`，请显式补齐对应 `SYSROOT`，或切换到可用的 `riscv64-elf-gcc` / `riscv64-unknown-elf-gcc`。

产物输出到 `build-riscv/` 和 `bin-riscv/`，与 Linux 本机构建完全隔离。`build-riscv/` 下会按 `scalar/` 和 `vector/` 分开保存目标文件，避免两套编译参数互相覆盖。

> 注意：在 x86 Linux 主机上执行 `Makefile.riscv` 时只会生成 RISC-V 可执行文件；训练和验证需要把生成的程序放到 RISC-V 环境中运行。

## 可执行文件

- `bin/train`：训练模型并导出参数头文件
- `bin/verify`：随机验证一张测试图片
- `bin-riscv/train-scalar.elf`：RISC-V 标量版训练程序
- `bin-riscv/verify-scalar.elf`：RISC-V 标量版验证程序
- `bin-riscv/train-vector.elf`：RISC-V 向量版训练程序
- `bin-riscv/verify-vector.elf`：RISC-V 向量版验证程序

## 清理

```bash
make clean
make -f Makefile.riscv clean
```

## 说明

- 首次运行前请先执行 `make data`
- `verify` 依赖 `src/model_params.h` 中的已训练参数，因此需要先执行 `make train`
- 如需调整训练轮数、batch size、学习率或随机种子，请编辑 `src/config.h`
- 默认目标是“项目尽量精简且好用”，因此实现上保持为最小可维护版本
- `Makefile.riscv` 是独立扩展入口，用于保留现有 Makefile 的同时提供 RISC-V 交叉编译能力
- RISC-V 版本会在 `train` 与 `verify` 的 `main` 前后读取 `instret` CSR，并打印 `instret delta (main)`
