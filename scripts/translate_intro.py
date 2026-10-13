from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'src/ml-engineering'
def write(path,text):
 p=ROOT/'content'/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.strip()+'\n')
write('training/README.md','''# 训练

**子章节**：

- [模型并行](model-parallelism)
- [性能](performance)
- [容错](fault-tolerance)
- [可复现性](reproducibility)
- [训练不稳定性](instabilities)
- [检查点](checkpoints)
- [训练超参数与模型初始化](hparams.md)
- [张量精度／数据类型](dtype.md)
- [仅用单节点模拟多节点环境](emulate-multi-node.md)——介绍如何在一台机器上模拟多节点环境；这里使用 `deepspeed` 启动器。
- [利用微调示例从头训练 HF Hub 上的模型](re-train-hub-models.md)
- [数据集](datasets.md)

**工具**：

- [printflock.py](tools/printflock.py)——一个小型库，让多 GPU 环境中的 `print` 调用输出不再交错。
- [multi-gpu-non-interleaved-print.py](tools/multi-gpu-non-interleaved-print.py)——基于 `flock` 封装 `print`，防止多个进程同时打印时消息相互交错；在多 GPU 上使用 `torch.distributed` 就会遇到这种情况。
''')
write('training/checkpoints/README.md','''# 检查点

- [torch-checkpoint-convert-to-bf16](./torch-checkpoint-convert-to-bf16)——把已有的 fp32 PyTorch 检查点转换为 bf16。如果发现 [safetensors](https://github.com/safetensors/safetensors) 文件，也会一并转换。这个工具应当很容易改造成其他类似用途。

- [torch-checkpoint-shrink.py](./torch-checkpoint-shrink.py)——有些检查点在保存时，由于某种原因，张量底层 storage 的大小超过了当前视图实际使用的大小。这个脚本通过克隆当前视图并重新保存，让检查点只保留当前视图所需的存储空间。
''')
write('training/hparams.md','''# 选择训练超参数与模型初始化方式

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
''')
# Preserve every fenced code block verbatim.
def interleave(path,parts):
 original=(SRC/path).read_text()
 codes=re.findall(r'```[^\n]*\n.*?```',original,re.S)
 assert len(parts)==len(codes)+1,(path,len(parts),len(codes))
 write(path,''.join(p+(codes[i] if i<len(codes) else '') for i,p in enumerate(parts)))
interleave('training/datasets.md',[
'''# 处理数据集

## 在主进程中预处理和缓存数据集

HF Accelerate 提供了非常方便的上下文管理器 [`main_process_first`](https://huggingface.co/docs/accelerate/v0.4.0/accelerator.html#accelerate.Accelerator.main_process_first)，可以这样编写代码：

''',
'''

从而代替下面这种不够直观、需要重复编写代码的方式：

''',
'''

应该在主进程中下载和处理数据，而不是让所有进程都执行这些操作。否则，各进程会并行重复相同的工作，而且很可能写入同一个位置，导致输出相互交错并损坏。从 I/O 的角度来看，串行执行这类工作通常也快得多。

这里有 `main_process_first` 和 `local_main_process_first` 两种上下文管理器。前者适用于数据位于共享文件系统、所有计算节点都能访问的情况；后者适用于数据保存在每个节点本地的情况。

如果没有使用 HF Accelerate，我也实现了类似的上下文管理器，只是名称不同：

- `global_main_process_first`：用于共享文件系统。
- `local_main_process_first`：用于节点本地文件系统。

可以在[这里](tools/main_process_first.py)找到它们。

如果希望编写能自动适配共享和本地文件系统的通用代码，该怎么办？我还增加了一个辅助函数，自动识别所使用的文件系统类型，并据此调用合适的上下文管理器。它叫作 `main_process_by_path_first`，用法如下：

''',
'''

可以在[这里](tools/main_process_first.py)找到它。

除了上下文管理器，当然还需要判断主进程类型的工具函数。因此，也提供了三个与上述管理器对应的函数：

- `is_main_process_by_path(path)`
- `is_local_main_process()`
- `is_global_main_process()`

这些函数都在[这里](tools/main_process_first.py)。

运行下面的命令即可查看它们的实际效果：

''','\n'])
interleave('training/re-train-hub-models.md',[
'''# 利用微调示例从头训练 HF Hub 上的模型

HF Transformers 提供了出色的微调示例：https://github.com/huggingface/transformers/tree/main/examples/pytorch 。这些示例几乎涵盖所有模态，而且开箱即用。

**但如果想从头训练，而不是微调，该怎么办？**

下面介绍一个简单的变通方法。

这里以 `facebook/opt-1.3b` 为例，并计划采用 bf16 训练：

''',
'''\n\n然后运行：\n\n''',
'''

这会创建 `opt-1.3b-bf16` 文件夹，包含从头训练模型所需的一切。换句话说，你得到了一个形式上类似预训练模型的模型，但它只完成了初始化，还没有进行任何训练。

如果计划使用 `torch.float16` 或 `torch.float32`，相应修改上面的脚本即可。

现在可以像平常微调一样，继续训练这个已经保存的模型：

''',
'''\n\n其中关键参数是：\n\n''',
'''

这里的 `opt-1.3b-bf16` 就是上一步生成的本地目录。

有时可以找到原模型训练时使用的相同数据集，有时则必须使用替代数据集。

其余超参数通常可以在模型随附的论文或文档中找到。

总之，这种方法让你能够利用微调示例，从头训练 [HF Hub](https://huggingface.co/models) 上找到的模型。
'''])
