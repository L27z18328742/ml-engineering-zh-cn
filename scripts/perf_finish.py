from pathlib import Path
import re
R=Path(__file__).resolve().parents[1]; p='training/performance/README.md'
s=''.join((R/'src/ml-engineering'/p).read_text().splitlines(True)[800:]); codes=re.findall(r'```[^\n]*\n.*?```',s,re.S)
parts=[
'''这个启动器目前需要辅助工具 [numa-set.sh](benchmarks/numa/numa-set.sh) 来设置 NUMA 亲和性。下载并添加执行权限后，可以这样设置：''',
'''注意：系统必须安装 `numactl` 才能使用该工具。

例如，可以这样验证分配是否正确：''',
'''在具有 208 个超线程 CPU 逻辑核的系统上，通常会看到：''',
'''前 4 个加速器使用前一半 CPU 核，另外 4 个使用后一半，符合前面介绍的正确配置。

移除 `./numa-set.sh` 后：''',
'''会发现全部 8 个进程都能看到全部 208 个 CPU 核：''',
'''每个进程都可以使用任意 CPU 核，因此可能发生跨 NUMA 节点访问，带来少量性能开销。

#### os.sched_setaffinity

当然，也可以在程序启动后修改 NUMA 亲和性。前面用 `os.sched_getaffinity` 读取当前设置，对应的 `os.sched_setaffinity` 则用于修改：''',
'''这里告诉系统，运行此脚本的进程（`0`）只能使用 CPU 核 `0` 和 `1`。

接下来只需以编程方式确定每个加速器进程对应的正确 CPU 核集合。[pynvml](#pynvml) 可以完成这件事。

#### pynvml

使用 NVIDIA GPU 时，`pynvml` 很适合查询 GPU 的各种信息，无须调用 `nvidia-smi`。安装命令是 `pip install nvidia-ml-py`，导入的模块名仍为 `pynvml`。这里用它根据 GPU 索引获取正确的 CPU 亲和性。

[numa-set-pynvml.py](benchmarks/numa/numa-set-pynvml.py) 提供了可用的辅助函数，可以在训练循环最前面调用：''',
'''务必在初始化 `DataLoader` **之前**调用，才能让 worker 使用正确的 CPU 核！

通常本地进程 rank 等于 GPU 索引，但设置 `CUDA_VISIBLE_DEVICES` 后可能不再成立。这时需要把进程 rank 映射为实际 GPU 索引：''',
'''另一个需要留意的设置是 `CUDA_DEVICE_ORDER`。它通常默认为 `PCI_BUS_ID`，混用不同 GPU 时也可能设为 `CUDA_DEVICE_ORDER=FASTEST_FIRST`。不过高端服务器配置几乎不会遇到这种情况，通常可以忽略。

#### srun

如果使用 SLURM，并愿意用 `srun` 启动，而不是 `torchrun`、`accelerate` 等，它会自动完成所有绑定工作。完整启动脚本见[这里](../../orchestration/slurm/launchers/srun-launcher.slurm)。

只需添加以下两项头部配置，即可启用 NUMA 亲和性：''',
'''`--ntasks-per-socket=4` 假设机器有 2 个 CPU 插槽和 8 个加速器，即每个插槽对应 `8/2=4` 个加速器。

这种方案更精确：每个进程都分到自己的 CPU 核组，而不仅是让驱动加速器 0–3 的进程共用 NUMA 节点 0 的所有核、驱动加速器 4–7 的进程共用 NUMA 节点 1 的所有核。

#### 不同启动器的支持

不同启动器对 NUMA 亲和性的支持如下：

- [HF Accelerate](https://github.com/huggingface/accelerate)：在 `accelerate` 启动命令中添加 `--enable_cpu_affinity` 即可，要求 `accelerate>0.28.0`。
- [torchrun](https://github.com/pytorch/pytorch)：没有内置支持，但可以采用[上面的方法](#numactl)。
- srun：见[上文](#srun)。

## DataLoader

### 异步 DataLoader

默认设置为 `num_workers=0`。这意味着每次调用 `next(iter(dataloader))` 时才实时读取数据，因此会产生 I/O 开销；如有数据变换，也要当场执行。这些工作都会阻塞加速器计算。

解决办法是设置 `num_workers > 0`，使用异步 `DataLoader`。除非 `DataLoader` 特别慢，通常 2 个 worker 就够了：''',
'''这样调用 `next(iter(dataloader))` 时，数据应该已经完成所有变换，并位于 CPU 内存中。仍然需要把它复制到加速器显存；如何加速这一步，参见[页锁定内存与非阻塞设备复制](#pinned-memory-and-non-blocking-device-copy)。

下面的基准测试模拟较慢的数据变换：[num-workers-bench.py](benchmarks/dataloader/num-workers-bench.py)。''',
'''在这个具体例子里，worker 数增加到 3 之前，速度都在明显提升。如果 `DataLoader` 很轻、工作很少，差异会小得多，但 0 个 worker 的开销总是最大。

可以测量实际工作负载，尝试更小或更大的值来调优。但要记住，每个 worker 都可能消耗大量 CPU 内存。一个 8 加速器节点若每个进程有 2 个 worker，就会额外产生 16 个进程。如今计算节点往往有数百个 CPU 核和数 TB 内存，可以支持很多 worker；过去则不同。

因为数据变换提前异步执行，这种情况下 CPU 和内存速度的影响没那么大。例如使用 2 个 worker 时，只要下一轮的数据准备耗时小于两轮计算耗时，`DataLoader` 就不应成为瓶颈。

注意，`num_workers > 0` 有时也会出问题。PyTorch 的一些相关 issue 多年未解决，表现为 worker 挂起，尤其在使用两个 `DataLoader` 时。BLOOM-176B 训练就遇到过：训练 `DataLoader` 使用 2 个 worker 一切正常，但加入评估 `DataLoader` 后会随机挂起。排查无果后，我们只对评估使用 `num_workers=0`，并大幅降低评估频率。最终完全停止同步评估，改为对保存的中间检查点做 lm-harness 风格的异步评估，也加快了训练。

案例：IDEFICS-80B 训练使用的流式 `DataLoader` 表现很差，每个 worker 都消耗大量内存，而且经常出现峰值。即便约有 1 TiB CPU 内存，也无法启动足够多的 worker，因此 `DataLoader` 成了瓶颈。当时没有时间寻找更好的方案，只能用它完成训练。

### 页锁定内存与非阻塞设备复制

结合以下设置，通常可以缓解 `DataLoader` 瓶颈：

1. `DataLoader(pin_memory=True, ...)`
2. `batch.to(device="cuda", non_blocking=True)`

其中：

1. 页锁定内存使 CPU 到加速器显存的数据传输更高效。
2. 非阻塞复制允许计算与数据移动部分重叠，进一步加速。

[pin-memory-non-block-bench.py](benchmarks/dataloader/pin-memory-non-block-bench.py) 是一个展示差异的小型基准测试。我在 A100 80GB-PCIe 上得到：''',
'''可见，`pin_memory=True` 配合 `non_blocking=True` 是值得尝试的改动。

更多背景见 [PyTorch 文档](https://docs.pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers)和 [NVIDIA 文章](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)。

注意：

- 操作系统会特殊处理页锁定内存，即使可用内存不足，也不会将其换出。因此它减少了其他程序可用的内存，使用时需要留意。
- 我记得有人数次报告使用页锁定内存后遇到问题；主要可能是系统本来 CPU 内存就少，或锁定了过多内存，导致操作系统大量使用交换空间。
- 测量[每轮迭代 TFLOPS](#tflops-as-a-performance-metric)，比较修改前后的吞吐量，选择效果最好的配置。若把 `DataLoader` 开销与前向／反向以及计算后的工作分别计时，会更容易看清影响。计算后的工作通常是日志记录，有时慢得令人意外。

## torch.compile

`torch.compile` 将模型跟踪成计算图，并生成融合 kernel，从而加速训练和推理。它随 PyTorch 2.0 发布；截至 2026 年 8 月，生态中已有很多项目默认使用它。但对任意模型获得良好加速仍不是自动发生的：跟踪器必须能理解你的代码，工作量也主要在这里。

### 在哪些地方有收益

回看[模型运算剖析](#anatomy-of-models-operations)中的三类运算，它们在编译下的表现差别很大：

1. **张量缩并**已经调用接近硬件极限的厂商 GEMM 库，编译器可改善的空间很小。`mode="max-autotune"` 会搜索 Triton 矩阵乘变体，偶尔可以超过厂商库，但这是最不容易获得收益的一类。
2. **统计归一化**和第三类**逐元素运算**才是主要收益来源。它们受内存带宽而非计算能力限制：从 HBM 读取输入，做少量运算，再写回结果。Eager 模式每个算子都要经历一次读写；融合为一个 kernel 后，整条运算链只需经历一次。这是 `torch.compile` 优化 Transformer 的主要方式，也解释了为什么大量小算子的模型比大矩阵乘主导的模型获益更多。

因此，可期待的加速取决于每步耗时中第二、三类工作的占比。先用 [PyTorch 性能分析器](../../debug/pytorch.md#profilers)分析，再编译。

如果瓶颈在 kernel 启动而不是带宽，还有另一种收益：`mode="reduce-overhead"` 将一步执行捕获为 CUDA graph，然后回放整个启动序列，避免分别发起数千次 kernel 启动。这对小批量和低延迟推理解码尤其重要，因为此时加速器可能闲等 CPU 派发任务。它会额外占用内存，因为捕获的图需要固定缓冲区；也要求静态形状。

2026-08-09 在单张 NVIDIA B200、`torch=2.13.0+cu130` 上的两组测量，展示了收益范围。两组中的 `forward` 行都是 `no_grad` 下的推理；另外两行涉及反向传播，均开启梯度检查点。这是在该规模下的常见做法，也使 8B 模型的一步计算能装入单卡显存。

第一组是短序列下的小模型：Llama 3.2 1B（`NousResearch/Llama-3.2-1B`），bf16、SDPA，批量 1、序列长度 512。每步只有数毫秒，主要受 kernel 启动开销和第二、三类运算的带宽限制，最有利于发挥 `torch.compile` 的优势。下表为预热后的稳态步耗时，括号内是相对 eager 的加速比：

| 工作负载 | eager | `torch.compile()` | `mode="reduce-overhead"` | `mode="max-autotune"` |
| :--- | ---: | ---: | ---: | ---: |
| 前向（推理） | 8.1ms | 3.4ms (2.4x) | 1.9ms (4.2x) | 2.0ms (4.1x) |
| 前向＋反向 | 48.8ms | 12.6ms (3.9x) | 8.1ms (6.0x) | 8.0ms (6.1x) |
| 再加 AdamW 更新 | 58.8ms | 18.7ms (3.1x) | 17.6ms (3.3x) | 17.5ms (3.4x) |

第二组是长序列下的大模型：Llama 3.1 8B（`meta-llama/Llama-3.1-8B`），相同数据类型与注意力后端，批量 1、序列长度 8192。每步数百毫秒，几乎全部时间都花在大型矩阵乘上，即第一类运算；厂商 GEMM 库已经让它们接近硬件极限：

| 工作负载 | eager | `torch.compile()` | `mode="reduce-overhead"` | `mode="max-autotune"` |
| :--- | ---: | ---: | ---: | ---: |
| 前向（推理） | 152.4ms | 155.6ms (0.98x) | 155.8ms (0.98x) | 154.8ms (0.98x) |
| 前向＋反向 | 590.7ms | 630.2ms (0.94x) | 629.2ms (0.94x) | 633.7ms (0.93x) |
| 再加 AdamW 更新 | 637.3ms | 673.8ms (0.95x) | 671.4ms (0.95x) | 680.0ms (0.94x) |

两表的差别说明了关键问题：受开销限制的小模型最多获得约 6 倍加速；计算受限的大模型没有收益，各种编译模式都与 eager 相差几个百分点，而且这里略慢。编译没有失败，只是没有多少可消除的开销。

1B 模型叠加了两种效果。首先，普通一步执行受启动和带宽限制，融合许多小 kernel 本身就很有效。其次，两行含反向传播的测试开启了梯度检查点：丢弃激活，在反向阶段重算整个前向，因此每步运行两次前向。在小模型上，额外前向同样受启动限制；编译器可以像融合原始 kernel 一样融合重算 kernel，吸收掉大部分额外开销，所以前向＋反向和加上 AdamW 两行的加速比大于仅前向。

8B 模型在该序列长度下没有这两种效应：GEMM 已接近 cuBLAS 峰值并占据主导，检查点重算只是增加更多同类矩阵乘；面对半秒量级的计算，编译能消除的启动间隙微不足道。因此三种编译模式都落在约 0.94 倍：`max-autotune` 搜索的 Triton 矩阵乘无法在这些形状上超过 cuBLAS，`reduce-overhead` 的 CUDA graph 消除的启动开销也已可忽略。

预期加速取决于一步执行受带宽或启动开销限制的程度，这正是[性能分析](#where-it-pays-off)要回答的问题：大量小算子的模型，或以小批量、短序列运行的模型，通常收益最大；大形状的密集矩阵乘负载收益最小，有时反而变慢。

冷启动成本真实存在；稳态收益很小或没有收益时，它决定了编译是否值得。使用全新 Inductor 缓存，默认编译的首次前向＋反向约为 22 秒（1B）和 47 秒（8B）；`max-autotune` 则增至 83 秒和 168 秒。1B 模型很快能收回成本：默认编译每步节省约 35ms，约 600 步后抵消初始成本，此后都是收益。8B／8192 配置没有可摊销的稳态收益，编译后反而略慢，因此编译时间完全是成本，在这种形状下无法收回；对于仅十步的冒烟测试尤为不利。

优化器更新不会增加这项编译成本：成本在于编译模型。已有前向／反向图后，把 `opt.step()` 放入计时区间不会重新编译。同一次运行先执行前向＋反向测试、建立图，再测试加 AdamW 的情况，因此后者冷启动列接近零。PyTorch 各版本的编译性能也在改进，所以绝对毫秒数以及 8B 结果的正负，应视为该工具链时点的结果，而非模型的永久特性。

两张表都来自 [torch-compile-bench.py](benchmarks/torch-compile-bench.py)。它遍历工作负载和编译模式，每个单元格创建新模型，对涉及反向的负载启用梯度检查点；将首步记为冷启动成本、后续步骤均值记为稳态时间，并逐模式输出相对 eager 的比率。模型、序列长度、批量、种子和模式列表都在文件顶部设置（`SEED = 42`）。默认配置得到上面的 8B 结果；取消 `NousResearch/Llama-3.2-1B` 和 `SEQ = 512` 两行注释即可得到 1B 表，批量仍为 1。种子固定随机输入 token，使重复运行使用相同数据。

### 难点在哪里

跟踪器要把 `forward` 转为图，无法跟踪的部分会产生**图中断（graph break）**：能编译的部分先编译，不能跟踪的部分回退到 eager，然后重新开始一个图。每个中断也是融合边界，因此存在很多中断的模型即使能在 `torch.compile` 下运行，也可能几乎没有收益。依赖数据的控制流、`.item()`、`print` 和不受支持的操作都会导致图中断。

另一个陷阱是**重新编译**。编译图针对跟踪时的形状特化，因此新序列长度会触发重编译。重编译次数受 `torch._dynamo.config.recompile_limit` 限制，默认 8 次；超过后编译器放弃该函数，之后直接使用 eager。常见原因是变长批次。如果输入形状变化，通常应使用 `torch.compile(model, dynamic=True)`：提前编译形状通用的图，牺牲一些峰值性能，以避免反复重编译。形状变化时应默认考虑它，不必等发现重编译才补救。

这两种问题都可以打印出来：''',
'''日志会把每个图中断对应到触发它的代码行，把每次重编译对应到失败的 guard。`torch.compile(model, fullgraph=True)` 会将图中断升级为错误，是快速找出它们的方法。

首次执行需要花时间编译，任何使缓存失效的变化也会再次带来编译成本。若模型由重复 `n` 次的同一种块组成，编译单个块而非整个模型可以显著降低成本：编译器只做一次工作，各层复用结果。

如果尝试后仍不工作：

1. 可以向 PyTorch 团队报告，最好提供最小可复现示例。
2. 可以按 PyTorch 官方的 [torch.compile 故障排查指南](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html)逐项排查。该指南与编译器同步维护，涵盖上述各种失败模式；有些问题可以自行解决，有些仍需要反馈给 PyTorch。

使用较新的 PyTorch 版本；编译器在版本间改进显著，旧版本的问题可能已在当前稳定版解决。

## 自动垃圾回收

Python 根据内部启发式规则定期自动进行垃圾回收。在数百至数千个加速器同步执行的 LLM 训练中，即使各 rank 代码路径只有细微差别，也可能在不同时刻触发垃圾回收。执行回收的一个或多个 rank 会变慢，成为拖慢整个集群的慢进程。

通常可以从 [MFU 曲线](#mfu-vs-hfu)中的向下尖峰看出这种现象。

若训练遇到此问题，可以在训练器开头禁用自动垃圾回收：''',
'''然后按所需间隔手动回收，例如每轮训练迭代调用一次：''',
'''更多细节见 [`gc` 文档](https://docs.python.org/3/library/gc.html)。

## 内存分析工具

本章讨论了各种功能理论上应消耗多少 MB 内存，但实际情况往往不同。你可能已规划好模型规模和批量，实际运行却突然内存不足。这时需要检查真实代码和模型，确定各部分分别占用多少内存，找出计算错误或漏算的额外开销。

为此可以使用内存分析器，现有工具有很多。

我开发的 [IPyExperiments](https://github.com/stas00/ipyexperiments) 可以方便地对每行或每块代码做快速分析。只需把代码放入 Jupyter notebook，它就会自动报告每个代码块分配或释放多少 CPU／GPU 内存。例如，可以查看加载模型用了多少内存，以及单次推理又额外占用了多少，并获得峰值内存报告。
''']
assert len(parts)==len(codes)+1,(len(parts),len(codes))
with (R/'content'/p).open('a') as f:
 for i,x in enumerate(parts):
  f.write('\n\n'+x.strip()+'\n')
  if i<len(codes): f.write('\n'+codes[i]+'\n')
