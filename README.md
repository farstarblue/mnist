# MNIST-C

一个尽量精简的纯 C99 MNIST 手写数字识别项目。

## 特性

- 纯 C 实现，不依赖任何第三方库
- 精简 CNN：`3x3 Conv(8) -> ReLU -> FC(5408 -> 10)`
- 卷积层使用 `ReLU`，输出层 `Softmax`，损失函数为交叉熵
- 小批量 SGD 训练
- He 初始化，正态分布使用 Box-Muller 生成
- 训练参数统一放在头文件 `src/config.h`
- `make data` 自动下载并校验 MNIST 数据集
- 训练完成后自动将 CNN 参数导出到头文件 `src/model_params.h`
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

训练结束后会生成 `src/model_params.h`，随后 `verify` 会自动使用这份 CNN 参数。
训练过程中会在终端按 batch 刷新当前 epoch 的进度和 batch loss。

### 4. 验证与基准

```bash
make verify
```

程序会：

- 从测试集中随机挑选一张图片
- 在终端打印字符画
- 输出真实标签和预测标签
- 显示 `CORRECT` 或 `WRONG`

也可以指定测试集中的固定样本下标：

```bash
./bin/verify --sample 0
./bin/verify --sample 1234
```

这里的“固定样本”表示本次运行固定选择测试集中的第 `INDEX` 张图片，而不是把程序永久写死成只验证某一张图。

如果想直接评估一段测试集上的准确率，可以指定从 `0` 到 `INDEX` 的所有图片：

```bash
./bin/verify --num 999
```

这个模式下不会打印字符画和单张图片预测结果，只会输出验证范围、样本数和准确率。

如需做更稳定的 RISC-V scalar/vector 对比，可以对同一张图片重复推理多次：

```bash
./bin/verify --sample 0 --repeat 1000
```

说明：

- `--sample INDEX` 固定验证测试集中的第 `INDEX` 张图片
- `--repeat COUNT` 对同一张图片重复推理 `COUNT` 次
- `--num INDEX` 验证测试集从 `0` 到 `INDEX` 的全部图片，并输出准确率
- 不传 `--sample` 时，默认仍然随机抽取一张图片
- `--num` 不能与 `--sample` 或 `--repeat` 同时使用
- RISC-V 版本的 `instret` 统计范围已缩小到 `network_predict()`，更适合比较推理路径的 RVV 加速效果

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
- 4 个 RISC-V ELF 默认都会静态链接，便于直接通过 `spike pk` 加载
- vector 版本在 `src/network.c` 中显式包含 `riscv_vector.h`，并对卷积/全连接点积热点使用 RVV intrinsics，因此会生成真实的 RVV 指令，而不是仅依赖编译器自动向量化

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

如果使用 `spike pk`，可直接运行静态产物，例如：

```bash
spike pk bin-riscv/verify-scalar.elf
spike pk bin-riscv/verify-vector.elf
spike pk bin-riscv/train-scalar.elf
spike pk bin-riscv/train-vector.elf
```

如需对比同一张图片在标量版和向量版上的指令数，可分别运行：

```bash
spike pk bin-riscv/verify-scalar.elf --sample 0 --repeat 1000
spike pk bin-riscv/verify-vector.elf --sample 0 --repeat 1000
```

如需对比一段测试集上的准确率，也可以分别运行：

```bash
spike pk bin-riscv/verify-scalar.elf --num 999
spike pk bin-riscv/verify-vector.elf --num 999
```

## 可执行文件

- `bin/train`：训练模型并导出参数头文件
- `bin/verify`：验证测试图片，支持随机抽样、固定样本或区间准确率评估
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
- `verify` 依赖 `src/model_params.h` 中的已训练 CNN 参数，因此需要先执行 `make train`
- 仓库默认提交的 `src/model_params.h` 只是占位头文件，用于保证新结构可编译；执行训练后会被真实参数覆盖
- 如需调整训练轮数、batch size、学习率或随机种子，请编辑 `src/config.h`
- 默认目标是“项目尽量精简且好用”，因此实现上保持为最小可维护版本
- `Makefile.riscv` 是独立扩展入口，用于保留现有 Makefile 的同时提供 RISC-V 交叉编译能力
- `verify` 支持 `--sample INDEX` 和 `--repeat COUNT`，便于固定样本并重复推理做基准测试
- `verify` 支持 `--num INDEX`，可对测试集 `0..INDEX` 批量评估并输出准确率
- RISC-V 版本的 `verify` 会在 `network_predict()` 前后读取 `instret` CSR，并打印单次或平均推理指令数
