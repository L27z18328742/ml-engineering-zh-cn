# 选择训练超参数与模型初始化方式

寻找一组适合作为起点的超参数和模型初始化设置，最简单的办法是借用一个已经成功的类似训练任务的配置。这里整理了一份[公开的 LLM／VLM 训练日志合集](../resources/README.md#publicly-available-training-llmvlm-logbooks)，可以作为起点。另一个常见来源是论文，前提是论文披露了这些信息。如果作者没有公开，也可以尝试联系他们，询问具体设置。

## 术语表

训练领域使用大量缩写和术语，下面列出本章涉及的一些重要术语。

- BS：Batch Size，批量大小。这里指每块 GPU 的批量大小，通常也称为 MBS（micro-batch-size，微批量大小）。
- GAS：Gradient Accumulation Steps，梯度累积步数；完成一次完整迭代前，需要执行多少轮前向／反向传播。
- GBS：Global Batch Size，全局批量大小；每次迭代的总批量大小，可能包括梯度累积。
- PP：Pipeline Parallelism，流水线并行。
- TFLOPS：每秒一万亿次浮点运算，参见 [FLOPS](https://en.wikipedia.org/wiki/FLOPS)。

## 逐步增大全局批量大小

如果计划使用很大的 GBS 进行训练，例如 1024、2048 个样本，甚至更多，那么一开始就向模型输入这么大的批量会非常浪费。此时模型还是完全随机的，无法从如此精细的数据统计中获益。因此，为了节省数据和资源，通常会在一段时间内逐步增大全局批量大小。

同样重要的是，初始 GBS 也不能太小，否则训练推进的效率不高。数据太少时，计算吞吐量（TFLOPS）较低，会拖慢整个训练过程。使用流水线并行（PP）时尤其如此，因为 PP 调优最重要的一点是缩小 GPU 空闲的气泡，而 GBS 越小，气泡越大。

例如，训练 BLOOM-176B 时我们使用了 PP。吞吐量基准测试发现，从 GBS=16 开始训练极慢（8 TFLOPS），因此最终从 GBS=192（73 TFLOPS）开始，然后逐步增加到 GBS=2048（150 TFLOPS）：每处理 9_765_625 个样本，就把 GBS 增加 16。

### 初始化标准差（STD Init）

这个超参数极其重要，要正确设置它需要进行数学推导。详情参见 [STD Init](instabilities/README.md#std-init)。
