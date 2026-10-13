from translate_intro import interleave
interleave('training/instabilities/README.md',[
'''# 避免、恢复与理解训练不稳定性

子章节：

* [理解训练损失曲线的形态](training-loss-patterns.md)——尖峰、发散、顿悟时刻、恢复训练等情况。

## 从训练日志中学习

最好的学习方式是阅读[公开的 LLM／VLM 训练日志](../../resources/README.md#publicly-available-training-llmvlm-logbooks)，因为其中可以看到具体发生了什么，以及问题是如何解决的。

## 初始化标准差（STD Init）

正确设置张量的初始分布，会对训练稳定性产生巨大影响。`std` 的取值并不是固定的，而是取决于隐藏维度的大小。

在 BLOOM 正式训练前的 104B 实验中，这被证明是一个至关重要的设置。我们一直无法顺利越过最初几千次迭代，直到发现 Megatron-LM 默认的 `--init-method-std=0.02` 对我们的模型来说太大了。

我们参考了两个来源：

1. “Transformers without Tears” 论文 https://arxiv.org/abs/1910.05895 建议使用：`sqrt(2/(NHIDDEN*5))`。
2. 530B 模型的训练论文 https://arxiv.org/abs/2201.11990 使用了更小的初始化公式：`sqrt(1/(NHIDDEN*3))`。

最终决定采用 530B 论文的公式，因为它会得到更小的初始值。

为了便于比较，这两个公式可以改写为：
1. `sqrt(0.4000/NHIDDEN)`
2. `sqrt(0.3333/NHIDDEN)`

因此，对于 `NHIDDEN=14336`，计算结果是 `sqrt(1/(14336*3)) = 0.00482`，我们使用的就是这个值。这当然不是 BLOOM-176B 训练没有出现稳定性问题的唯一原因，但我认为它是关键原因之一。

## 数值不稳定性

调试章节中的[检测异常张量值](../../debug/pytorch.md#detecting-problematic-tensor-values)和[下溢与上溢检测](../../debug/pytorch.md#underflow-and-overflow-detection)介绍了定位 `inf`／`nan` 值的工具，也值得参考。

处理低精度数值时，某些数学运算可能不稳定。

例如，可以阅读这份很有启发性的 [PyTorch 数值稳定性指南](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)。

下面用一个具体例子说明。

在采用 fp16 混合精度的 104B 训练实验中，[Corby Rosset](https://github.com/corbyrosset) 提出了一项改进，让[自注意力更加稳定](https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/118)。

具体来说，这[一行代码](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/c839a8aa30731f71b3738d56009be9668508e366/megatron/model/transformer.py#L303)显示，`norm_factor` 可能在 Query × Key 矩阵乘法完成后才乘上去。如果 Q 和 K 的维度非常大，输出可能已经溢出，此时再使用 `norm_factor` 也无法补救。

建议把 `norm_factor` 移入运算内部，在矩阵乘法之前先缩小 Q 和 K：

''',
'''

为了让变换前后的运算在数学上等价，将归一化因子移入内部时，需要再取一次平方根。若 n 是标量，A 和 B 是矩阵，则：

''',
'''

这样，A 和 B 的维度就可以显著增大。

同样的“先乘法、后缩放”问题，在更底层的 CUDA 矩阵乘法 API 中也存在。

对 CUDA 内核开发者而言，在本文写作时，[CuBlas](https://docs.nvidia.com/cuda/cublas/index.html) 的 `GemmStridedBatchedEx` 也有类似问题。它的定义是：

''',
'''

问题在于，`alpha` 是在矩阵乘法完成之后才乘上去的，因此可能引起不稳定。

## 数据批次与模型参数状态的“不良”组合

PaLM 团队在训练较大模型时，观察到数十次损失尖峰，出现的间隔“极不规律”。虽然未能定位根因，但他们通过从更早的检查点重启、跳过可能有问题的数据批次来缓解问题。参见[第 5.1 节：训练不稳定性](https://arxiv.org/pdf/2204.02311)。

## Adam 中由时间域相关性引起的发散

[A Theory on Adam Instability in Large-Scale Machine Learning](https://arxiv.org/abs/2304.09871) 对训练最高达 546B 参数的 LLM 时出现的发散尖峰进行了严谨研究，认为时间域相关性会导致 Adam 发散。当 epsilon 不够小、梯度估计中的分量变得与 epsilon 大小接近时，就会触发这种问题。

论文第 7.1 节提出了一些实践建议，其中最值得注意的是将 epsilon 设为 0，并酌情处理除零情况。
'''])
