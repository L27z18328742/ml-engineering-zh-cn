# 模型并行

## 并行概览

现代机器学习使用各种并行方法来实现以下目标：

1. 突破 GPU 显存限制，例如：
   - 容纳非常大的模型：t5-11b 仅模型参数就占 45 GB。
   - 容纳非常长的序列：例如 Llama 3.1 模型的 128K 上下文长度。
2. 显著加速训练：将原本需要一年的训练缩短到数小时。

我们会先深入讨论各种一维并行技术及其优缺点，再介绍如何组合成二维、三维并行，以进一步加速训练、支持更大的模型。还会介绍其他一些强大的替代方法。

虽然主要概念大多也适用于其他框架，本文重点讨论基于 PyTorch 的实现。

对于大小超过加速器显存容量的模型，训练和推理主要采用两种方法：
1. 三维并行：网络通信效率很高，但可能对模型代码有很强的侵入性，需要投入更多工作才能正确运行。
2. ZeRO 并行：网络通信效率不高，但几乎不需要修改模型代码，很容易用起来。

## 扩展性相关概念

下面简要介绍本文后续会深入讨论的主要概念。

1. [数据并行](#data-parallelism)（DP）：将相同配置复制多份，每份输入不同的数据切片。各副本并行处理，并在每个训练步结束时同步。
2. [张量并行](#tensor-parallelism)（TP）：将每个张量拆成多个块。整个张量不再位于单块 GPU 上，而是每个分片放在指定 GPU 上。前向和反向计算时，各分片并发执行，并在特定层边界交换部分激活或梯度，常用 all-reduce、all-gather 或 reduce-scatter，而不是只在训练步结束时同步。因为切分发生在水平方向，也可以称为水平并行。参见 [PyTorch 张量并行教程](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)。
3. [流水线并行](#pipeline-parallelism)（PP）：沿垂直方向、按层将模型分到多块 GPU 上，每块 GPU 只放一层或若干层。各 GPU 并行处理流水线的不同阶段，每次处理批量中的一小块。
4. [零冗余优化器](#zero-data-parallelism)（ZeRO）：也会对张量分片，与 TP 有些相似，但会在前向或反向计算需要时及时重建完整张量，因此不需要修改模型。它还支持各种卸载技术，以弥补 GPU 显存不足。其他 ZeRO 实现中，常用 Sharded DDP 指代这一基础概念。
5. [序列并行](#sequence-parallelism)：长输入序列训练需要大量 GPU 显存，这种技术将单个序列的处理分散到多块 GPU 上。
6. [专家并行](#expert-parallelism)：混合专家模型（MoE）可以进行划分，让每个专家拥有一块或若干块专属 GPU。

在我读过的资料中，[Breadth-First Pipeline Parallelism](https://arxiv.org/abs/2211.05953) 论文的引言部分，可能是对常见并行技术解释得最好的资料之一。

## 数据并行

### DDP

即使只有两块 GPU，大多数用户也已经通过几乎开箱即用的 `DataParallel`（DP）和 `DistributedDataParallel`（DDP）获得了训练加速。这是 PyTorch 内置的功能。

详情参见 [DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)。

### ZeRO 数据并行

下面这张来自[博客文章](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)的图，展示了基于 ZeRO 的数据并行（ZeRO-DP）：

![DeepSpeed 图 1](images/parallelism-zero.png)

初看可能难以理解，但概念其实很简单。它就是普通的 `DataParallel`（DP），区别在于不再复制完整模型参数、梯度和优化器状态，而是每块 GPU 只保存一部分。运行时，当某一层需要完整参数，所有 GPU 就同步交换彼此缺少的部分，仅此而已。

考虑一个简单模型，共 3 层，每层 3 个参数：

```
La | Lb | Lc
---|----|---
a0 | b0 | c0
a1 | b1 | c1
a2 | b2 | c2
```

La 层的权重为 a0、a1 和 a2。

如果有 3 块 GPU，Sharded DDP（即 ZeRO-DP）会这样划分模型：

```
GPU0:
La | Lb | Lc
---|----|---
a0 | b0 | c0

GPU1:
La | Lb | Lc
---|----|---
a1 | b1 | c1

GPU2:
La | Lb | Lc
---|----|---
a2 | b2 | c2
```

想象常见的 DNN 结构图，这在某种程度上与张量并行一样，属于水平切分。垂直切分则是把完整的一组层放到不同 GPU 上。不过，这还只是开始。

接下来，每块 GPU 都像普通 DP 一样获得一个小批量：

```
x0 => GPU0
x1 => GPU1
x2 => GPU2
```

输入没有改变，仍像是要交给普通模型处理一样。

首先，输入到达 La 层。

只看 GPU0：x0 的前向计算需要 a0、a1、a2，但 GPU0 只有 a0。因此，它从 GPU1 接收 a1，从 GPU2 接收 a2，将所有参数拼齐。

与此同时，GPU1 收到小批量 x1，自身只有 a1，还需要 a0 和 a2，于是分别从 GPU0、GPU2 获取它们。

收到 x2 的 GPU2 也做同样的事：从 GPU0、GPU1 获取 a0、a1，再加上自身的 a2，重建完整张量。

三块 GPU 都重建完整张量，然后执行前向计算。

计算一结束，就丢弃不再需要的数据；它们只在计算期间使用。通过预取，可以高效完成重建。

整个过程依次在前向的 Lb、Lc 层重复，然后在反向传播中按 Lc → Lb → La 的顺序重复。

在我看来，这就像一种高效的团队徒步负重分配策略：

1. A 带帐篷。
2. B 带炉子。
3. C 带斧头。

每晚大家共享自己的装备，并向其他人借用自己没有的东西；早晨各自收起分配给自己的装备，继续前进。这就是 Sharded DDP／ZeRO DP。

相比之下，如果每个人都要带自己的帐篷、炉子和斧头，就低效得多。这对应 PyTorch 的 DataParallel，也就是 DP 和 DDP。

阅读相关文献时，你可能会遇到 Sharded、Partitioned 这些同义用语。

仔细观察 ZeRO 划分模型权重的方式，会发现它与后面介绍的张量并行很相似，因为它对每层的权重进行分区／分片，而不是像接下来介绍的垂直模型并行那样分层。

ZeRO-DP 第 1、2、3 阶段的实现：
- [DeepSpeed](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch](https://docs.pytorch.org/docs/stable/fsdp.html)：最初在 [FairScale](https://github.com/facebookresearch/fairscale/) 中实现，后来合入 PyTorch 核心。
- [torchtitan](https://github.com/pytorch/torchtitan)

DeepSpeed ZeRO 的集成：
- [HF Trainer 集成](https://huggingface.co/docs/transformers/main_classes/deepspeed)
- [Accelerate](https://huggingface.co/docs/accelerate/usage_guides/deepspeed)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/deepspeed.html)
- [Determined.AI](https://github.com/determined-ai/determined/tree/main/docs/model-dev-guide/api-guides/apis-howto/deepspeed)

FSDP 的集成：
- [HF Trainer 集成](https://huggingface.co/docs/transformers/main/en/fsdp)
- [Accelerate](https://huggingface.co/docs/accelerate/main/en/usage_guides/fsdp)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html)
- [torchtitan](https://github.com/pytorch/torchtitan)

重要论文：

DeepSpeed 与 ZeRO：
- [ZeRO：面向万亿参数模型训练的内存优化](https://arxiv.org/abs/1910.02054)
- [ZeRO-Offload：让十亿级模型训练更加普及](https://arxiv.org/abs/2101.06840)
- [ZeRO-Infinity：突破超大规模深度学习的 GPU 显存墙](https://arxiv.org/abs/2104.07857)
- [ZeRO++：面向巨型模型训练的高效集体通信](https://arxiv.org/abs/2306.10209)
- [DeepSpeed Ulysses：支持超长序列 Transformer 模型训练的系统优化](https://arxiv.org/abs/2309.14509)
- [AMSP：降低 ZeRO 的通信开销，实现高效 LLM 训练](https://arxiv.org/abs/2311.00257)

PyTorch：
- [PyTorch FSDP：扩展全分片数据并行的经验](https://arxiv.org/abs/2304.11277)

DeepSpeed ZeRO 的主要资源：
- [项目 GitHub](https://github.com/deepspeedai/DeepSpeed)
- [使用文档](https://www.deepspeed.ai/getting-started/)
- [API 文档](https://deepspeed.readthedocs.io/en/latest/index.html)
- [博客文章](https://www.microsoft.com/en-us/research/search/?q=deepspeed)

#### 解决全局批量过大的问题

例如使用 1024 个加速器时，每个加速器只保存很小的分片，剩余大量显存可以用于微批量（MBS）。假设能容纳 MBS=32，最后就会得到 GBS=32k，这很可能不是你想要的。

因此，要么采用实现起来并不简单的[张量并行](#tensor-parallelism)，要么采用通常更容易部署的[序列并行](#sequence-parallelism)。我还没有实际尝试，但目前了解到的方案是：

- DeepSpeed ZeRO 使用 [DeepSpeed-Ulysses](#deepspeed-ulysses-sp)。
- [Arctic 长序列训练](#arctic-long-sequence-training)。
- FSDP 使用 [Paged Ring Attention](https://github.com/lucidrains/ring-attention-pytorch)（[论文](https://arxiv.org/abs/2402.08268)）。

请注意，它很可能没有[张量并行](#tensor-parallelism)高效，但我目前还不知道额外开销具体是多少。

#### 具有多个副本的 ZeRO

默认情况下，ZeRO 使用所有 GPU 构成一个模型副本，也就是把模型分散到全部 GPU 上。这会产生一些限制：

1. 全局批量大小不够灵活，始终是 `total_gpus*micro_batch_size` 的函数。大集群上可能得到极大的全局批量，不利于高效收敛。当然，可以用很小的微批量控制全局批量，但每块 GPU 上的矩阵会变小，计算效率也随之降低。
2. 无法充分利用快得多的节点内网络，因为较慢的节点间网络决定了整体通信速度。

[ZeRO++](https://arxiv.org/abs/2306.10209) 通过引入 ZeRO 分层权重分区（hpZ）解决第二个限制。这种方法不再把整个模型的权重分散到全部 GPU，而是将每个模型副本限制在单个节点内。内存总占用因此增加到原来的节点数倍，但用于收集分片的两次 `all_gather` 都能通过快得多的节点内连接执行。只有聚合并重新分发梯度的 `reduce_scatter` 通过较慢的节点间网络执行。

第一个限制并没有真正解决，因为整体全局批量仍然相同。不过，每个副本效率更高，而且额外显存压力可能限制每块 GPU 可用的微批量，因此总体上应该能提高系统吞吐量。

PyTorch FSDP 通过 [shardingStrategy.HYBRID_SHARD](https://docs.pytorch.org/docs/stable/fsdp.html) 实现了这项功能。

论文：

- [ZeRO++：面向巨型模型训练的高效集体通信](https://arxiv.org/abs/2306.10209)
- [PyTorch FSDP：扩展全分片数据并行的经验](https://arxiv.org/abs/2304.11277)

#### ZeRO 的变体

以下已发表论文提出了对 ZeRO 机制的修改：

- [MiCS：在公有云上以接近线性的扩展性训练巨型模型](https://arxiv.org/abs/2205.00119)（2022）
- [AMSP：通过先进的模型状态分区实现 LLM 训练的超线性扩展](https://arxiv.org/abs/2311.00257)（2023）

## 流水线并行方法

### 朴素模型并行（垂直）

朴素模型并行（MP）将模型的一组组层分布到多块 GPU 上。机制相对简单：通过 `.to()` 将指定层放到指定设备；数据进入或离开这些层时，将数据移到与层相同的设备，其余部分保持不变。

我们称它为垂直 MP，因为按照大多数模型结构图的画法，切分方向是垂直的。例如，下面是一个 8 层模型：

```
===================  ===================
|  0 | 1 | 2 | 3  |  |  4 | 5 | 6 | 7  |
===================  ===================
        gpu0                 gpu1
```

我们将它垂直切成两部分，0–3 层放到 GPU0，4–7 层放到 GPU1。

数据从第 0 层到第 1 层、从第 1 层到第 2 层、再从第 2 层到第 3 层时，与普通模型完全一样。但从第 3 层传到第 4 层时，需要从 GPU0 传输到 GPU1，因此增加了通信开销。如果这些 GPU 位于同一计算节点，也就是同一台物理机器，复制通常很快；如果位于不同计算节点，也就是多台机器，通信开销可能显著增加。

之后，第 4→5→6→7 层与普通模型一样。第 7 层完成后，通常需要将数据传回存放标签的第 0 层，或者反过来将标签发送到最后一层。这样才能计算损失，并让优化器完成工作。

问题：
- 主要缺点，也是它被称为“朴素”MP 的原因，是任意时刻都只有一块 GPU 在工作，其余 GPU 都空闲。因此，使用 4 块 GPU 几乎只是把单块 GPU 的显存容量扩大到四倍，却没有利用其余计算硬件，还增加了设备间复制的开销。4 块 6 GB 显卡用于分层状态时，总容量大致相当于 1 块 24 GB 显卡，但单卡不需要设备间传输，训练更快。朴素 MP 主要是增加容量的技术：当与各层相关的训练状态超过单卡显存时，只要每个分区及其激活能装入对应 GPU，就能运行这项工作负载。
- 共享嵌入可能需要在 GPU 之间反复复制。

### 流水线并行

流水线并行（PP）与朴素 MP 几乎相同，但它会将输入批量拆成微批量，人为构建流水线，让不同 GPU 同时参与计算，从而解决 GPU 空闲问题。

下面这张来自 [GPipe 论文介绍](https://research.google/blog/introducing-gpipe-an-open-source-library-for-efficiently-training-large-scale-neural-network-models/)的图，上半部分是朴素 MP，下半部分是 PP：

![模型并行与流水线并行](images/parallelism-gpipe-bubble.png)

从下图很容易看出，PP 中 GPU 空闲的区域更少。这些空闲部分称为“气泡”。

图中两种方式的并行度都是 4，即 4 块 GPU 参与流水线。前向依次经过 F0、F1、F2、F3 四个流水线阶段，然后反向以 B3、B2、B1、B0 的顺序返回。

PP 引入了一个需要调节的新超参数 `chunks`，定义有多少个数据块会依次经过同一个流水线阶段。例如，下图中 `chunks=4`。GPU0 对第 0、1、2、3 块依次执行相同的前向计算（F0,0、F0,1、F0,2、F0,3），然后等待其他 GPU 工作。直到它们开始完成任务，GPU0 才重新开始，对第 3、2、1、0 块执行反向计算（B0,3、B0,2、B0,1、B0,0）。

注意，在概念上，这与梯度累积步数（GAS）相同。PyTorch 称之为 `chunks`，DeepSpeed 则将同一个超参数称为 GAS。

数据分块让 PP 引入了微批量（MBS）的概念。DP 把全局数据批量拆成小批量：如果 DP 并行度为 4，全局批量 1024 就会分成 4 个各含 256 个样本的小批量（1024/4）。如果 `chunks` 或 GAS 为 32，那么微批量大小就是 8（256/32）。每个流水线阶段一次处理一个微批量。

因此，DP + PP 的全局批量大小为 `mbs*chunks*dp_degree`，即 `8*32*4=1024`。

再回到图中。

当 `chunks=1` 时，就退化为非常低效的朴素 MP。而 `chunks` 很大时，微批量又会变得太小，同样可能低效。因此，需要实验寻找能让 GPU 利用率最高的取值。

图中可以看到，有些“死时间”气泡无法并行消除，因为最后的前向阶段必须等待反向过程完成流水线。寻找最佳 `chunks` 值的目标，就是让所有参与 GPU 都保持较高的并发利用率，也就是尽量缩小气泡。

调度策略对性能至关重要。按提出的先后顺序，常见策略包括：

- 顺序调度：[GPipe：用流水线并行高效训练巨型神经网络](https://arxiv.org/abs/1811.06965)。
- 交错 1F1B：[PipeDream：快速高效的流水线并行 DNN 训练](https://arxiv.org/abs/1806.03377)。
- 循环、深度优先：[使用 Megatron-LM 在 GPU 集群上高效训练大规模语言模型](https://arxiv.org/abs/2104.04473)。
- 广度优先：[Breadth-First Pipeline Parallelism](https://arxiv.org/abs/2211.05953)。
- Llama 3 训练结合深度优先和广度优先来获得最佳性能，还能随着训练推进逐步调整全局批量大小；这通常很难在 PP 中实现。参见 [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) 第 3.3.2 节“用于模型扩展的并行”。

下面是一个交错流水线的例子：

![交错流水线执行](images/parallelism-sagemaker-interleaved-pipeline.png)

这里通过优先执行反向传播，进一步缩小气泡，也就是空闲时间。

DeepSpeed 和 SageMaker 等系统采用了这种方式。

[DeepSeek v3](https://arxiv.org/abs/2412.19437) 通过 DualPipe 引入了效率更高的 PP，缩小气泡，并实现更好的计算与通信重叠。具体细节参见论文第 3.2.1 节。

![DualPipe](images/parallelism-pp-dualpipe.png)
（[来源](https://arxiv.org/abs/2412.19437)）

### PP 的实际使用限制

无论使用哪种 PP 技术栈，都会反复遇到几个约束：

- 模型需要较大修改，因为普通模块图必须切分成流水线阶段。一些 API 仍要求类似由多个阶段组成的 `nn.Sequential`，可能需要改变模型设计。
- 阶段边界能够传递的内容有限。如果单进程前向中有一组 Python 变量流经第一阶段，就必须想办法适配。典型接口要求阶段输入和输出是单个 Tensor 或 Tensor 元组，而且张量第一维必须是批量维度，因为流水线要把小批量拆成微批量。
- 无法在流水线阶段层面执行条件控制流。例如，T5 这样的编码器—解码器模型，需要特殊处理带条件的编码器阶段。
- 必须安排好各层，使一个阶段的输出成为下一阶段的输入。
- 第一阶段包含较重的嵌入层。词表很大时，这一层也会很大，可能需要自定义切分，让第一阶段包含的 Transformer 块少于其他阶段。

现代技术栈减少了一些样板代码，但没有消除这些结构约束。

### 当前有哪些 PP 实现

实际可选的是自带 PP 的生产训练技术栈，以及 PyTorch 的可组合 API：

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)：经过大量实际训练验证的 PP，支持交错 1F1B 等调度；与模型密切绑定，没有独立公开的 Pipe 风格 API。
- [DeepSpeed](https://www.deepspeed.ai/tutorials/pipeline/)：PP 是 DeepSpeed 技术栈的一部分，经常与 ZeRO 结合。
- [torch.distributed.pipelining](https://docs.pytorch.org/docs/stable/distributed.pipelining.html)：PyTorch 当前的 PP 工具集，由 [PiPPy](https://github.com/pytorch/pippy) 迁入；仍处于 **alpha** 阶段，API 可能变化。可以切分通用模型，提供 GPipe、1F1B、交错 1F1B、循环 BFS 和 DeepSeek 风格的 DualPipeV 等调度，并能与 DP／FSDP／TP 组合。三维并行 Llama 示例参见 [torchtitan](https://github.com/pytorch/torchtitan)。
- [torchtitan](https://github.com/pytorch/torchtitan)／[nanotron](https://github.com/huggingface/nanotron)：采用现代 PP 的完整训练技术栈。
- [DeepSeek DualPipe](https://github.com/deepseek-ai/DualPipe)：来自 [DeepSeek-V3](https://arxiv.org/abs/2412.19437) 的调度策略，在 `torch.distributed.pipelining` 中也以 `ScheduleDualPipeV` 提供。
- [SageMaker](https://arxiv.org/abs/2111.05972)：专有方案，仅用于 AWS。

### 相关阅读

- [流水线并行：通过模型划分进行分布式训练](https://siboehm.com/articles/22/pipeline-parallel-training)

## 张量并行

张量并行中，每块 GPU 只处理张量的一部分，只有在操作需要完整张量时才聚合它。

本节使用 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 论文[《在 GPU 集群上高效训练大规模语言模型》](https://arxiv.org/abs/2104.04473)中的概念和图示。

Transformer 的主要构建单元是全连接 `nn.Linear`，后接非线性激活 `GeLU`。

按照 Megatron 论文的记号，可以将这一计算写为 `Y = GeLU(XA)`，其中 `X`、`Y` 为输入和输出向量，`A` 为权重矩阵。

以矩阵形式来看，很容易理解如何在多块 GPU 之间分配矩阵乘法：

![并行 GEMM](images/parallelism-tp-parallel_gemm.png)

如果沿列方向将权重矩阵 `A` 划分到 `N` 块 GPU，并行计算 `XA_1` 到 `XA_n`，就会得到 `N` 个输出向量 `Y_1, Y_2, ..., Y_n`，它们可以分别独立送入 `GeLU`：

![独立的 GeLU](images/parallelism-tp-independent-gelu.png)

利用这一原理，可以更新任意深度的 MLP，而无需在 GPU 之间同步，直到最后需要从分片重建输出向量。Megatron-LM 论文作者给出了清晰的图示：

![并行分片处理](images/parallelism-tp-parallel_shard_processing.png)

多头注意力层更容易并行化，因为它本来就包含多个相互独立的头，天然具有并行性！

![并行自注意力](images/parallelism-tp-parallel_self_attention.png)

重要：TP 要求非常快的网络。通常节点内网络比节点间网络快得多，因此不建议跨节点使用 TP。实际中，如果一个节点只有 4 块 GPU，最高 TP 并行度就是 4；如果需要 TP=8，就应该使用至少有 8 块 GPU 的节点。

TP 可以与其他并行方法结合。

其他名称：
- DeepSpeed 称其为[张量切片（tensor slicing）](https://www.deepspeed.ai/tutorials/large-models-w-deepspeed/)。

实现：
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)：与具体模型密切相关，因此采用内部实现。
- [PyTorch](https://docs.pytorch.org/docs/stable/distributed.tensor.parallel.html)
- [SageMaker](https://arxiv.org/abs/2111.05972)：专有方案，只能在 AWS 使用。
- [OSLO](https://github.com/eleutherAI/Oslo)：基于 Transformers 实现张量并行。
- [nanotron](https://github.com/huggingface/nanotron)
- [torchtitan](https://github.com/pytorch/torchtitan)

### 异步 TP

TP 的一个缺点是通信难以与计算重叠。PyTorch 提出了 [Async-TP](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487)，把具有依赖关系的 all-gather + matmul 序列拆成一系列 cudaMemcpyAsync 调用和较小的局部矩阵乘法，借此解决问题，而且可以通过 `torch.compile` 自动完成！

- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) 也通过 `--tp-comm-overlap` 实现了这项功能。

### 相关阅读

- [张量并行与序列并行：详细分析](https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/#sequence-parallelism)

## 序列并行

DNA 测序等机器学习任务可能需要非常长的序列，例如 256K；即使普通 LLM，也可能使用 10k 甚至更长的序列进行训练。

自注意力是 Transformer 的关键组件。对于标准稠密注意力，计算复杂度随序列长度呈平方增长。显式构造完整“序列×序列”注意力矩阵的内核，还需要平方级临时显存。[FlashAttention](https://arxiv.org/abs/2205.14135) 等节省显存的内核避免保存这个矩阵，让注意力显存占用与序列长度呈线性关系，但没有消除平方级计算量。

序列足够长时，即使批量大小为 1，也可能无法装入单块 GPU。因此，序列并行沿序列维度划分工作。它扩大了可支持的序列长度，但并不是无限长度：实际限制仍然取决于加速器总显存、计算时间、通信开销和内核约束。

这种并行方式与本文介绍的其他并行类型正交，因此可以与它们组合，形成四维并行、ZeRO-DP+SP 等方案。

### DeepSpeed-Ulysses SP

论文：[DeepSpeed Ulysses：支持超长序列 Transformer 模型训练的系统优化](https://arxiv.org/abs/2309.14509)

这一实现对两类内容进行分片：
1. 多头注意力权重分配到参与的 GPU，每块 GPU 只拥有若干子头。这个步骤发生在模型创建／加载时，与[张量并行](#tensor-parallelism)有些相似。
2. 训练时，每条输入序列分成多个块，各块发送到一块 GPU。这让人想到 ZeRO-3 分片，但这里分的是输入，而不是权重。

计算时，每个序列块先投影为 QKV，然后在每块设备上收集为完整序列的 QKV。每个设备只计算其拥有的子头，之后再次收集为完整注意力输出，交给 MLP 块。

![DeepSpeed-Ulysses 序列并行](images/deepspeed-ulysses.png)

[来源](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-ulysses)

图中步骤如下：
1. 输入序列 N 划分到 P 个可用设备。
2. 每个本地 N/P 输入分区投影为查询（Q）、键（K）和值（V）嵌入。
3. 然后通过参与设备间高度优化的 all-to-all 集体通信，将本地 QKV 嵌入收集为全局 QKV。
4. 接着分别执行各个头的注意力计算：

![计算公式](images/deepspeed-ulysses-math.png)

5. 最后，再执行一次 all-to-all，将注意力输出的上下文张量转换回沿序列划分的 N/P 形式，供 Transformer 层块中后续的 MLP 矩阵乘法、LayerNorm 等算子使用。

例如，设 seqlen=8K、num_heads=128，单节点 num_gpus=8：

1. 每块 GPU 得到原序列中长度为 1K 的一块（`8K/8`）。
2. 每块 GPU 分配 16 个子头（`128/8`）。
3. a. 在 gpu0 的 `forward` 之前，将原序列收集回 8K 个 token。
   b. 在前 16 个子头上执行注意力计算。

其余 7 块 GPU 执行相同逻辑，各自在自己的 16 个子头上计算长度为 8K 的注意力。

这种高效通信的具体细节可以在[这里](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-ulysses#significant-communication-volume-reduction)阅读。

DeepSpeed-Ulysses 通过让 GPU 数量与消息大小或序列长度成比例增长，保持通信量不变。

### Arctic 长序列训练

Arctic Long Sequence Training 将 [DeepSpeed-Ulysses](#deepspeed-ulysses-sp) 移植到 HuggingFace Transformers，并更新以支持现代注意力头机制。它还通过分块计算和卸载激活检查点进一步扩展能力，支持更长的序列或更大的批量。[这里](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/)是集成指南。

- 论文：https://arxiv.org/abs/2506.13996
- 实现与集成：[ArcticTraining](https://github.com/snowflakedb/ArcticTraining/tree/main/projects/sequence-parallelism) 和 [Axolotl](https://github.com/axolotl-ai-cloud/axolotl)

### Colossal-AI 的 SP

论文：[序列并行：从系统视角看长序列训练](https://arxiv.org/abs/2105.13120)

Colossal-AI 的 SP 实现使用环形自注意力。这是一种环状通信方式：query 投影留在本地，key 和 value 投影沿环传递，以计算全局注意力。通信复杂度与消息大小 M 呈线性关系。

### Megatron-LM 的 SP

论文：[减少大型 Transformer 模型中的激活重计算](https://arxiv.org/abs/2205.05198)

Megatron-LM 的 SP 与其 TP 紧密集成。它沿序列维度划分序列，通过 all-gather 和 reduce-scatter 集体通信聚合用于注意力计算的 QKV 投影。无论计算设备数量多少，其通信量都随消息大小 M 线性增加。

### 采用分块 Transformer 的 Ring Attention

论文：[Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889)

也称为 Ring Self-Attention（RSA，环形自注意力）。这个名称由 [Colossal-AI 的 SP](#colossal-ais-sp) 论文提出，描述的同样是 query 保留本地、key 和 value 沿环传递的方法。两个名称都很常见，因此 [DistFlashAttn](#distflashattn) 与 Ring Self-Attention 的对比，指的就是这一技术。

1. 张量始终沿序列维度分片，形状为 `(seq_len // N, d_model)`。
2. 在注意力层，每块 GPU 先用手头已有的分片，计算能够计算的那部分注意力分数。
3. 同时，其他序列块的 key 和 value 沿环传递。
4. 收到另一个块的 key／value 后，各 GPU 使用这一新序列片段的 key／value 张量继续计算注意力。
5. 持续进行，直到注意力计算完成。

SP 实现：
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [Colossal-AI](https://colossalai.org/)
- [torchtitan](https://github.com/pytorch/torchtitan)

PyTorch 也在开发这项功能，将其称为上下文并行（CP）。

### DistFlashAttn

据 [DISTFLASHATTN：面向长上下文 LLM 训练的分布式高效显存注意力](https://arxiv.org/abs/2310.03294) 报告，它比 Ring Self-Attention 快数倍，因为在执行序列并行时，它对各 worker 之间每个 token 的 KVQ 计算进行了负载均衡。

![DistFlashAttn](images/dist-flash-attn.png)

### 相关阅读

- [张量并行与序列并行：详细分析](https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/#sequence-parallelism)

## TP+SP

可以在同一进程组内结合 TP 和 SP，尽量降低通信成本，参见[减少大型 Transformer 模型中的激活重计算](https://arxiv.org/abs/2205.05198)。例如在 LLM 中，嵌入层、注意力层和线性层采用 TP，而到了 dropout 和 LayerNorm，则改用 SP。

## DP+PP

下面这张来自 DeepSpeed [流水线教程](https://www.deepspeed.ai/tutorials/pipeline/)的图，说明了如何组合 DP 与 PP。

![DP 与 PP 二维并行](images/parallelism-zero-dp-pp.png)

关键是注意：DP rank 0 看不到 GPU2，DP rank 1 看不到 GPU3。对 DP 来说，只有 GPU0 和 GPU1，它就像总共只有两块 GPU 那样分发数据。GPU0 “悄悄”通过 PP 将部分工作交给 GPU2；GPU1 也同样请 GPU3 帮忙。

因为每个维度至少需要两块 GPU，所以这里至少需要四块。

实现：
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [SageMaker](https://arxiv.org/abs/2111.05972)
- [OSLO](https://github.com/eleutherAI/Oslo)
- [nanotron](https://github.com/huggingface/nanotron)
- [torchtitan](https://github.com/pytorch/torchtitan)

## DP+PP+TP

为了进一步提高训练效率，可以组合 PP、TP 和 DP，形成三维并行，如下图所示。

![DP、PP、TP 三维并行](images/parallelism-deepspeed-3d.png)

图片来自博客[《三维并行：扩展到万亿参数模型》](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)，这篇文章也值得阅读。

每个维度至少需要两块 GPU，因此这里至少需要八块。

实现：
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed)：还提供更高效的 DP，称为 ZeRO-DP。
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- [SageMaker](https://arxiv.org/abs/2111.05972)
- [OSLO](https://github.com/eleutherAI/Oslo)
- [nanotron](https://github.com/huggingface/nanotron)
- [torchtitan](https://github.com/pytorch/torchtitan)

## ZeRO DP+PP+TP

DeepSpeed 的主要功能之一是 ZeRO，它是具有极强扩展能力的 DP 扩展，前面的 [ZeRO 数据并行](#zero-data-parallelism)已经介绍。通常它可以独立使用，不需要 PP 或 TP，但也可以与两者组合。

ZeRO-DP 与 PP 结合时，无论是否同时使用 TP，通常只启用 ZeRO 第 1 阶段，即优化器分片。

理论上，ZeRO 第 2 阶段，也就是梯度分片，可以与流水线并行一起使用，但会损害性能。每个微批量都需要额外执行一次 reduce-scatter，在分片前聚合梯度，可能增加显著通信开销。流水线并行本身使用较小的微批量，重点是在算术强度（微批量大小）与缩小流水线气泡（微批量数量）之间找到平衡，因此这些通信成本会带来负面影响。

此外，PP 已让每块 GPU 上的层数减少，进一步节省的显存也不会很多。PP 已将梯度大小缩小到 `1/PP`，因此在此基础上再做梯度分片，收益不如纯 DP 明显。

出于同样原因，ZeRO 第 3 阶段也不是好选择，因为需要更多节点间通信。

使用 ZeRO 的另一个好处是 ZeRO-Offload。这里使用的是第 1 阶段，因此可以将优化器状态卸载到 CPU。

实现：
- [Megatron-DeepSpeed](https://github.com/deepspeedai/Megatron-DeepSpeed) 和 [BigScience 的 Megatron-DeepSpeed](https://github.com/bigscience-workshop/Megatron-DeepSpeed)，后者是前者的分支。
- [OSLO](https://github.com/eleutherAI/Oslo)
- [torchtitan](https://github.com/pytorch/torchtitan)

重要论文：

- [使用 DeepSpeed 和 Megatron 训练大型生成语言模型 Megatron-Turing NLG 530B](https://arxiv.org/abs/2201.11990)

## 专家并行

使用混合专家模型（MoE），尤其在推理期间，可以给每个专家分配专属加速器；如果一块不够，也可以分配多块。这称为专家并行（EP）。它增加了一个并行维度，对很可能命中全部专家的大批量尤其能显著加速。EP 通信传输的是 token，而不是模型权重。矩阵乘法处理的输入更大，因此计算更高效。

详细解释请参见：
- [DeepSpeed-MoE：推进混合专家推理与训练，支撑下一代 AI 规模](https://arxiv.org/abs/2201.05596)
- [混合专家模型详解](https://huggingface.co/blog/moe#parallelism)

## 其他并行方法

### FlexFlow

[FlexFlow](https://github.com/flexflow/flexflow-train) 采用稍有不同的方式解决并行问题。

论文：[Zhihao Jia、Matei Zaharia、Alex Aiken 的《超越深度神经网络的数据并行与模型并行》](https://arxiv.org/abs/1807.05358)

它在 Sample、Operator、Attribute、Parameter 四个维度上进行一种四维并行：

1. Sample（样本）＝数据并行，沿样本划分。
2. Operator（算子）＝将单个操作并行拆成多个子操作。
3. Attribute（属性）＝数据并行，沿长度划分。
4. Parameter（参数）＝模型并行，不区分水平还是垂直。

示例：

* Sample

假设有 10 个长度为 512 的序列。沿样本维度分给两块设备，原来的 10×512 就变为 5×2×512。

* Operator

LayerNorm 需要均值和方差，因此标准差与均值并不是独立操作。为了演示如何把算子分解到两块设备（cuda:0、cuda:1），可以先将输入复制到两块设备，再让 cuda:0 归约 `sum(x)`，cuda:1 归约 `sum(x*x)`。结果组合为 `mean = E[x]` 和 `variance = E[x*x] - E[x]**2`。这种直接使用矩的公式可能损失数值精度，因此生产实现通常采用稳定的归约算法。参见 PyTorch 的 [`torch.std_mean` 文档](https://docs.pytorch.org/docs/stable/generated/torch.std_mean.html)。

* Attribute

还是 10 个长度为 512 的序列。沿属性维度分给两块设备，10×512 就变为 10×2×256。

* Parameter

类似张量模型并行或朴素的按层模型并行。

![FlexFlow SOAP](images/parallelism-flexflow.jpeg)

这个框架的意义在于，它综合考虑 GPU／TPU／CPU、RAM／DRAM、快速节点内连接／较慢节点间连接等资源，通过算法自动优化，决定在什么位置使用哪种并行方式。

一个重要前提是，FlexFlow 为工作负载静态且固定的 DNN 模型优化并行方式。对于具有动态行为的模型，不同迭代可能适合不同的并行策略。

因此，它的设想很有吸引力：在选定集群上运行 30 分钟模拟，就能得到充分利用该环境的最佳策略。添加、移除或替换任何组件后，它都会重新运行并优化方案，然后就可以训练。不同配置会有各自定制的优化方案。

## 并行中的网络集体通信

节点内和节点间速度相差一个数量级，因此，对节点内通信与跨节点通信选择不同并行技术至关重要。例如，TP 同步需求极高，应保持在速度最快、延迟最低的通信域内。传统上这意味着单节点内部，但 NVL72 这类机架级系统的通信域跨越多个节点，也称超节点。因此，真正的原则是让 TP 留在 scale-up 互联域内；只有实测表明能够承受，或容量限制别无选择时，才跨入较慢的 scale-out 网络。此外，某些加速器，例如较新的 AMD MI3\*\* 系列，GPU 间连接较慢，也会影响获得最佳性能所需的并行配置。

一个有用的知识点是：all-reduce 可以拆成 reduce-scatter 和 all-gather 两个独立阶段。

![all-reduce 等于 reduce-scatter 加 all-gather](images/all-reduce-reduce-scatter-all-gather.png)
（[来源](https://engineering.fb.com/2021/07/15/open-source/fsdp/attachment/fsdp-graph-2a/)）

各并行策略使用的集体通信如下：

- DDP：对梯度执行一次 all-reduce，理想情况下与计算重叠；总通信量约为模型参数量的 2 倍。
- ZeRO-DP ZeRO-1／ZeRO-2：对梯度执行一次 reduce-scatter，再执行一次 all-gather 收集更新后的所有参数；总通信量约为模型参数量的 2 倍。
- ZeRO-DP ZeRO-3：对权重执行两次 all-gather，分别位于 `forward` 和 `backward` 之前；对梯度执行一次 reduce-scatter。总通信量约为模型参数量的 3 倍，是 DDP 和 ZeRO-1／ZeRO-2 的 1.5 倍。
- TP：两次 all-gather 和两次 reduce-scatter。
- PP：两次 send 和两次 recv，在稳定阶段与计算重叠。
- SP：取决于实现。设隐藏维度为 h、序列长度为 N、并行度为 P：
  * Megatron-LM：两次 all-gather 和两次 reduce-scatter，每个 Transformer 层通信量为 `4*N*h`，参见[论文第 3.2 节](https://arxiv.org/abs/2309.14509)。
  * DeepSpeed Ulysses：两次 all-to-all，每个 Transformer 层通信量为 `4*N*h/P`，参见[论文第 3.2 节](https://arxiv.org/abs/2309.14509)。

不同实现可能采用不同通信模式。

## ZeRO 对节点间速度的要求

无论是 DeepSpeed ZeRO 还是 PyTorch FSDP，ZeRO 扩展机制所需的节点间流量，都比 TP+PP+DP 方案多得多。有时它也无法利用更快的节点内互联。因此，节点间网络较慢时，昂贵的 GPU 可能严重受限于通信。

ZeRO 会将一部分通信与计算重叠，理想目标是接近 `comms_time <= compute_time`。重叠并不完美，因此总会存在一些网络瓶颈，但应该确保 `comms_time` 不会远大于 `compute_time`。

ZeRO-3 在 `forward` 中对权重执行 `all_gather`，在 `backward` 中再次对权重执行 `all_gather`，最后在反向期间对梯度执行 `reduce_scatter`。总共三次全局集体通信，每次发送的数据量为模型参数数目乘以每参数字节数。例如，10B 参数模型采用 bf16 和 ZeRO-3，需要发送 `10*2*3`＝60 GB 数据。

相比之下，[DistributedDataParallel](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)（DDP）只调用一次 `all_reduce`，但它需要两倍的数据传输。因此，同一个 10B、bf16 模型采用 DDP 时，需要发送 `10*2*2`＝40 GB 数据。

只分片优化器状态的 ZeRO-1 与 DDP 一样，也需要传输 40 GB，即一次 `all_gather` 加一次 `reduce_scatter`。

通信和计算时间的估算公式如下，单位为秒：

- `comms_time = comms_multiplier * n_bytes * model_size_in_B / inter-node-throughput_in_GBps`
- `compute_time = n_passes * n_bytes * model_size_in_B * seqlen * global_batch_size / (total_gpus * 1e3 * tflops_wo_comms)`

计算时间公式是一种适用于基于 Transformer 块模型的粗略估计。它忽略小规模计算，只包含主要的矩阵乘法。

通信时间公式也是一阶估计。`comms_multiplier` 是单步中各次集体通信的 [busbw 修正因子](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bandwidth)之和：`all_reduce` 为 `2(n-1)/n`，`all_gather` 和 `reduce_scatter` 各为 `(n-1)/n`，其中 n 为 rank 数。这就是 DDP 的 2 和 ZeRO-3 的 3 的来源。公式省略了 `(n-1)/n`；rank 很多时是很好的近似，例如下面 512 个 rank 时为 0.998，但 rank 较少时就不够准确，8 个 rank 时为 0.875。代入的吞吐量必须是实测 `busbw`，而不是 `algbw`，因为 rank 增加时 `algbw` 会缩小。即使负载相同，不同集体通信的 `busbw` 也可能有所变化，因为 NCCL 可能为它们选择不同算法或协议。

下面使用 [IDEFICS-80B](https://huggingface.co/HuggingFaceM4/idefics-80b/) 的训练数据做个实验。

训练 IDEFICS-80B 时，我们使用 340 Gbps EFA，在 A100 上用 DeepSpeed ZeRO-3 只能得到 90 TFLOPS，而采用 Megatron 的 TP+PP+DP 可达到 150+ TFLOPS。此外，因为新模型是在一个语言模型和一个视觉模型的基础上构建的，有相当一部分模型被冻结，所以通信乘数本来小于 3。另一方面，我们使用激活重计算来节省显存，这增加了一次全部模型权重的传输；再加上当时 NCCL 不能正确支持半精度归约，我们用 fp32 归约梯度。因此，实际乘数不是 3，而更接近 4.5。

IDEFICS-80B 训练使用的取值：

- `model_size_in_B`＝`80`。
- `n_bytes`＝`2`，bf16 每个值占 2 字节。
- `comms_multiplier`：ZeRO-3／FSDP 为 `3`，即一次 reduce_scatter 加前向、反向各一次 all_gather；ZeRO-1 为 `2`，即一次 reduce_scatter 加一次 all_gather。
- 此外，IDEFICS-80B 为减少 NCCL 累加损失，选择[用 fp32 归约梯度](../dtype.md#when-to-use-fp32-accumulators)。增加的 2 字节使 `comms_multiplier*n_bytes=3*2+2=4*2`。但约一半模型被冻结，只需发送约一半梯度，因此乘数仍为 3。
- `n_passes`：使用激活重计算时为 `4`，不使用时为 `3`。一次前向对应一份计算，反向对应两份，因为分别要计算对输入、对权重的梯度。激活重计算再增加一次前向。
- `total_gpus`＝`512`。
- `global_batch_size`＝`3584`。
- `seqlen`＝`1024`。
- `inter-node-throughput_in_GBps`＝42.5，即 340 Gbps，使用 AWS EFA v1。
- `tflops_wo_comms` 是没有通信开销时的 TFLOPS。不是无法达到的理论峰值，而是例如 A100@BF16 理论值的约 80%，即 `312*0.8=~250` TFLOPS，向上取整。

我们用 4 GiB 负载运行 `all_reduce_bench.py`，测得节点间吞吐量 340 Gbps；该脚本现已改名为 [torch-dist-bench.py](../../network/benchmarks/torch-dist-bench.py)。IDEFICS-80B 有 80 层，每层约 1B 参数，也就是每层 bf16 张量发送 2 GB、fp32 张量发送 4 GB，与网络基准测试相符。如果层小得多，建议调整基准测试负载。例如，每层只有 100M 参数时，bf16 负载就是 0.2 GB。小了一个数量级，网络可达到的带宽很可能更低，计算时应使用对应数值。

补充说明：如果部分模型被冻结，同步梯度的数据量会减少。IDEFICS 冻结了超过一半的模型，因此梯度归约流量只有约一半。

代入得到：

- 通信＝`3 * 2 * 80 / 42.5`＝11 秒。
- 计算＝`4 * 2 * 80 * 1024 * 3584 / (512 * 1e3 * 250)`＝18 秒。

查看 IDEFICS-80B 日志，每次迭代约 49 秒。

好消息是，通信加计算的估算与实测处于同一量级。

还可以用日志中的 90 TFLOPS 代入计算公式进行合理性检查：

- 计算＝`4 * 2 * 80 * 1024 * 3584 / (512 * 1e3 * 90)`＝51 秒。

49 秒与 51 秒很接近。但这其实没有提供新的信息，因为日志中的 TFLOPS 原本就是由这个公式计算出来的，当然应该吻合。

最理想的情况是，将接近理论峰值的 TFLOPS 代入公式，得到的计算时间估计与系统实测计算时间接近。记住，通信与计算交错执行，因此测量 `forward`+`backward` 的墙钟时间时，其中也包含通信。

结论是什么？我认为还需要进一步调查，因为显然还有隐藏瓶颈。我已无法访问这个环境，等下一次训练较大模型时，会重新做这项分析，并分享更新的计算结果。不过，这个练习应该已经能帮助你理解后台发生的事情，以及这些数字之间如何关联。

上面的计算还没有纳入梯度累积步数 GAS，IDEFICS-80B 当时没有使用梯度累积。如果 GAS>1，理论计算时间不变，但通信时间从 `3*2*M/GBps` 变为 `GAS*3*2*M/GBps`。前向和反向所需的权重 `all_gather`，都会重复 GAS 次。理论上梯度只需通信一次，但由于每块 GPU 没有空间保存已收集权重的中间梯度，梯度也必须归约 GAS 次。这适用于 ZeRO-2 和 ZeRO-3；ZeRO-1 在 GAS>1 时不需要额外通信。

这里也没有讨论 `DataLoader` 这个潜在瓶颈，但我们测试过它用时不到 1 秒，开销很小。

通信估算还没有考虑各种硬件延迟，但负载很大时，这些延迟应该不会带来显著额外开销。

现在，你已经知道在自己的系统网络上传输这么多 GB 数据需要多长时间。例如，如果网络比 IDEFICS-80B 使用的网络慢 5 倍，即 8.5 GBps（68 Gbps）：

- 通信＝`3 * 2 * 80 / 8.5`＝56 秒。

与更快的计算相比，这肯定会成为严重瓶颈。

如果网络快 5 倍，即 212 GBps（1700 Gbps）：

- 通信＝`3 * 2 * 80 / 212`＝2 秒。

相对计算时间，这就很小了，尤其是其中一部分还能与计算成功重叠。

此外，DeepSpeed 团队在 384 块 V100、24 个 DGX-2 节点上对 [176B 模型进行了实测](https://github.com/deepspeedai/DeepSpeed/issues/2928#issuecomment-1463041491)，发现：

1. 100 Gbps IB：每块 GPU 不到 20 TFLOPS，较差。
2. 200–400 Gbps IB：每块 GPU 约 30–40 TFLOPS，尚可。
3. 800 Gbps IB：每块 GPU 超过 40 TFLOPS，优秀。

提醒一下，NVIDIA V100 的 fp16 峰值为 [125 TFLOPS](https://www.nvidia.com/en-gb/data-center/tesla-v100/)。

但要小心，这是 V100 的测试！其半精度计算比 A100 慢约 2–3 倍，比 H100 慢约 4–8 倍。因此，要让 H100 节点在半精度下达到上面类似的比例，通信至少要快 4–8 倍。我们需要更新硬件的更多测试。

补充说明：2–3 倍这个范围，是因为官方规格称 V100→A100、A100→H100 每一代 TFLOPS 都提高 3 倍，但用户实测报告的提升最多约为 2.5 倍。

他们还注意到，大规模训练时，每 GPU 的微批量越小，通信开销越明显。为了保证模型收敛，全局批量往往固定，因此可能无法增大微批量。新引入的 [ZeRO++](#zero-with-multiple-replicas) 解决了这一问题。

最后，进行上述计算时，必须知道自己环境的实际带宽。它随负载大小变化，负载越大，带宽通常越高。需要查看 DeepSpeed 配置中的 `reduce_bucket_size` 和 `prefetch_bucket_size`，分别对应归约和预取。默认值为 0.5B 参数，半精度时为 1 GB（0.5B×2 字节），fp32 时为 2 GB（0.5B×4 字节）。因此，应使用相应负载运行 `all_reduce` 基准测试，获取实际带宽，再代入上面的计算。

## 何时使用哪种策略

下面给出非常粗略的选择指南。每组列表排在前面的方案通常更快。

**⇨ 单 GPU**

* 模型能装入单块 GPU：

    1. 正常使用即可。

* 模型无法装入单块 GPU：

    1. ZeRO + CPU 卸载，必要时再使用 NVMe。
    2. 如果最大层也无法装入单块 GPU，在上述基础上增加 Memory Centric Tiling，详见下文。

* 最大层无法装入单块 GPU：

    1. ZeRO：启用[以显存为中心的分块（Memory Centric Tiling，MCT）](https://deepspeed.readthedocs.io/en/latest/zero3.html#memory-centric-tiling)。它自动拆分层并顺序执行，让任意大的层也能运行。MCT 减少同一时刻驻留 GPU 的参数数量，但不减少激活显存。截至 2026 年 8 月，这种需求仍很少见，因此需要用户手动替换 `torch.nn.Linear`。

**⇨ 单节点／多 GPU**

* 模型能装入单块 GPU：

    1. DDP，即分布式数据并行。
    2. ZeRO：是否更快取决于具体情形和配置。

* 模型无法装入单块 GPU：

    1. PP。
    2. ZeRO。
    3. TP。

    如果节点内有非常快的 NVLink 或 NVSwitch，三者性能应该大致相当；没有这些高速互联时，PP 通常比 TP 或 ZeRO 更快。TP 并行度也可能影响结果，最好在具体环境中实验比较。

    TP 几乎总是在单节点内使用，即 TP 大小不超过每节点 GPU 数。

* 最大层无法装入单块 GPU：

    1. 如果不使用 ZeRO，必须采用 TP，因为仅靠 PP 无法容纳。
    2. 使用 ZeRO 时，参见上面“单 GPU”的同类情况。

**⇨ 多节点／多 GPU**

* 如果模型能装入单节点，先尝试[多个副本的 ZeRO](#zero-with-multiple-replicas)：让 ZeRO 走更快的节点内连接，DDP 走较慢的节点间连接。

* 节点间连接较快时：

    1. ZeRO：几乎不需要修改模型。
    2. PP+TP+DP：通信更少，但需要大量修改模型。

* 节点间连接较慢，而且 GPU 显存仍不足时：

    1. DP+PP+TP+ZeRO-1。
