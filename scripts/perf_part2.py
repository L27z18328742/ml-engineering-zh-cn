from pathlib import Path
import re
R=Path(__file__).resolve().parents[1]; p='training/performance/README.md'
s=''.join((R/'src/ml-engineering'/p).read_text().splitlines(True)[250:520]);codes=re.findall(r'```[^\n]*\n.*?```',s,re.S)
parts=[
'''- 带动量 SGD、LION、Adafactor 等优化器：4 字节×参数数，只维护一份状态；Adafactor 在这 4 字节之外还需要一些额外内存。
- [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 等 8 位 AdamW：2 字节×参数数。

**梯度：**

- fp32，以及某些框架的混合半精度训练：4 字节×参数数。
- 非混合半精度，以及某些框架的混合半精度训练：2 字节×参数数。

**前向激活：**

- 大小取决于许多因素，主要是序列长度、隐藏维度和批量大小。

这里包含前向、反向函数传入和返回的输入输出，以及为梯度计算保存的前向激活。

每 GPU 激活显存可粗略拆为：

1. 激活工作显存：`dtype_size * num_of_hidden_states_copies * bs * seqlen * hidden_size`
2. 激活检查点显存：`dtype_size * num_layers * bs * seqlen * hidden_size`
3. logits 工作显存：`fp32_size * bs * seqlen * vocab_size`

以下计算中：
- `dtype_size`：数据类型单元素字节数，如 fp8 为 1、bf16 为 2、fp32 为 4，完整列表见[张量精度／数据类型](../dtype.md)。
- `fp32_size`：4 字节。
- `bs`：批量大小。
- `hidden_size`：模型隐藏维度。
- `seqlen`：序列长度。
- `num_layers`：层数。
- `vocab_size`：词表大小。

`dtype_size * bs * seqlen * hidden_size` 就是 `hidden_states` 张量大小，等于形状 `[bs, seqlen, hidden_size]` 乘以每元素字节数。

下面以 Llama-3 架构的 [8B 配置](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/config.json)为例：

''',
'''\n\n并使用：\n\n''',
'''

立即可得 bf16 `hidden_states` 大小：`2 * 1 * 32768 * 4096 / 2**30 = 0.25GiB`。

第 1 部分：计算单层前向所需工作显存：

`activation_memory_per_layer = num_of_hidden_states_copies * dtype_size * bs * seqlen * hidden_size`

`num_of_hidden_states_copies` 通常为 20–50，随架构变化。用 [activation-memory-per-layer.py](benchmarks/activation-memory-per-layer.py) 测几个模型，结果为：

''',
'''

可见，单层前向中 Gemma 所需 `hidden_states` 副本数是 SmolLM2 的两倍。

`hidden_size` 不同时，直接比较显存困难，但比较副本数系数很容易。

有些架构还分配其他内存，例如 `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` 的系数不是整数。

回到示例：

`activation_memory_per_layer = 2 * 28 * 1 * 32768 * 4096 / 2**30 = 7GiB`，28 来自上面 Llama-3.1-8B-Instruct 的测量。

第 2 部分：知道单层显存后，取决于是否启用[梯度检查点](#gradient-checkpointing)，有两种情况：

- 启用时，保存 `num_layers * hidden_states`。示例为 `32 * 0.25 = 8GiB`。
- 不启用时，激活显存增长到 `num_layers * activation_memory_per_layer`。示例为 `32 * 7 = 224GiB`。

8 GiB 对 224 GiB，差异极大，因此即使需要额外重算一次前向，大家仍会启用[梯度检查点](#gradient-checkpointing)。

第 3 部分：损失计算通常显式生成 fp32 logits，词表往往是最大维度，因此大小为 `fp32_size * bs * seqlen * vocab_size`。

示例为 `4 * 1 * 32768 * 128256 / 2**30 = 15.7GiB`，约 16 GiB。

logits 比该模型全部激活检查点占用还大。

实际中系数往往是 6 而不是 4，因为通常先生成低精度 logits，再转为 fp32，两份同时存在，即 `2+4=6`。

第 4 部分：启用检查点时，总激活显存为单层前向工作显存、激活检查点和 logits 之和；未启用时，为全部前向激活加 logits。

示例：

- 启用：`7 + 8 + 16 = 31GiB`。
- 未启用：`224 + 16 = 240GiB`。

**临时显存：**

各种临时变量计算后会释放，但峰值时也会额外占用，触发 OOM。因此写代码时应统筹考虑，必要时在不用后立即显式释放。

**特定功能显存：**

软件还可能有特殊需求，例如 beam search 生成文本时，需要保存多份输入输出。

**推理**的计算类似，但去掉优化器状态和梯度，权重只需参数数乘以：

- fp32：4 字节。
- fp16／bf16：2 字节。
- fp8／int8：1 字节。
- fp4／int4：0.5 字节。

fp8、fp6、fp4、整数量化和 MX 等格式，参见[推理章节的模型权重](../../inference/README.md#model-weights)。

[Transformer Math 101](https://blog.eleuther.ai/transformer-math/)也详细介绍显存及其他需求。

[EAI cookbook](https://github.com/EleutherAI/cookbook) 提供一组[计算脚本](https://github.com/EleutherAI/cookbook/tree/main/calc)，根据配置估计训练或推理的理论内存开销。

Alexander Smirnov 的 [GPU VRAM Estimator](https://vram.asmirnov.xyz/) 也很方便，另有[原理说明](https://asmirnov.xyz/vram)。

### 其他 GPU 显存占用

除上述部分外，还有其他显存使用者，因此模型永远无法使用全部标称显存。

#### 预加载 CUDA 内核的显存占用

PyTorch 首次使用 CUDA 时，可能占用 0.5–2 GiB，减少总可用显存，而且 torch 显存分析器不会统计这部分。

不同 GPU、PyTorch 版本占用不同。下面只创建一个 4 字节 CUDA 张量，观察初始化占用。

`pytorch==1.10.2`：

''',
'''\n\n`pytorch==1.13.1`：\n\n''',
'''

旧版在 A100 上“占掉”1.78 GiB，新版仅 0.9 GiB；省下约 0.9 GiB，可能正好避免 OOM。

但不要认为这是单向下降趋势。它取决于具体 PyTorch、CUDA 版本如何打包和加载内核，版本间可能增减：1.10.2→1.13.1 几乎减半，到下面的 2.1.1 又略升。这些 2022–2023 年数据保留在这里，是说明仅换版本，同 GPU 就可能相差近 1 GiB。它们只是现象示例，不应拿来规划容量；应在自己的 GPU、PyTorch、CUDA 组合上运行命令测量。

较新 PyTorch 中，`CUDA_MODULE_LOADING=EAGER` 强制预加载，否则按需延迟加载。生产中不要这样设置，因为可能加载不必要的内核，浪费显存；延迟加载的目的就是只加载需要的内核。

`pytorch==2.1.1`：

''',
'''\n\n相比之下，延迟加载模式：\n\n''',
'''

相差约 450 MiB，但这里只加载了 `torch.ones` 所需内核。真实程序调用更多 API 后，占用会位于 0.47–0.92 GiB 之间。

#### `torch.distributed` 的显存占用

使用 `torch.distributed`，仅初始化就可能占约 1–2 GiB；GPU 越多，占用越高，不同后端也不同，而且 torch 显存分析器不会统计它。

[torch-dist-mem-usage.py](distributed/torch-dist-mem-usage.py) 展示实际占用：

''',
'''

可见，[CUDA 内核](#preloaded-cuda-kernels-memory-usage)先占 0.55 GiB，第一次 `dist.barrier` 再占约 1.5 GiB。如果 `init_process_group` 传入 `device_id`，额外分配会移到初始化调用，barrier 的新增占用则减少。

#### 如何选择 `nccl`、`nccl2` 与 `nccl-lazy`

从 PyTorch 2.14 开始，分布式包在同一 NCCL 库之上提供第二套实现。`nccl2` 并不代表 NCCL 版本 2，也不替换 NCCL 集体通信算法；它改变的是 PyTorch 如何创建、拆分、等待和中止通信器。2.14 中 `init_process_group("nccl")` 仍选择原有 `ProcessGroupNCCL`。显式使用 `backend="nccl2"`，或设置 `TORCH_DIST_USE_NCCL2=1`，才会采用新实现。`nccl-legacy` 始终选择旧实现；`nccl-lazy` 是 nccl2 的变体，在首次访问某个 peer 时创建通信器，而非提前创建。

空闲 8×H200 节点上，PyTorch 2.14.0、CUDA 13.0、NCCL 2.30.7，1 GiB `all_reduce` 两者 `busbw` 都在 458–466 GBps。同一实现重复测试的波动与两者差距一样大，因此没有显示吞吐量差异，它们调用的 NCCL 集体通信相同。

可测的差异在显存：初始化和首次 barrier 后，rank 0 在 nccl 下占 1.56 GiB，在 nccl2 下占 5.16 GiB。多次复现，是否传入 `device_id` 都一样。因此，不要为期待更快通信而切换，采用前应测自己的拓扑显存成本。nccl2 提供的是通信器生命周期控制：`new_group` 时即建立子组，而非等首次通信；还可通过 2.14 标为不稳定的 API，在 rank 故障后原地重建组，无需重启。

`nccl-lazy` 适合每个 rank 只与少量 peer 通信的情况，例如[流水线并行](../../training/model-parallelism/README.md#pipeline-parallelism)，从不使用的通信器就不创建。如果进程组从第一步起就执行集体通信，通信器最终都要建立，因此不会降低峰值显存。

参考：[PyTorch 2.14 发布博客](https://pytorch.org/blog/pytorch-2-14-release-blog/)。

#### 显存碎片

模型不断分配和释放张量，可能产生碎片。例如总空闲量足够分配连续 1 GiB，但分散在数百个小片段中，除了很小的分配外无法使用。

`PYTORCH_ALLOC_CONF` 可以调整默认分配机制，详情见[内存管理](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management)。当代码频繁改变张量形状、容易严重碎片化时，我发现 `PYTORCH_ALLOC_CONF=expandable_segments:True` 非常有帮助。

### 批量大小

通常有两种批量：

1. 微批量 MBS，也称每 GPU 批量，是单 GPU 在一次 `forward` 中处理的样本数。
2. 全局批量 GBS，是所有参与 GPU 在两次优化器更新之间总共处理的样本数。

一个模型副本需要多少 GPU，取决于装下完整模型所需资源。

- 模型能装入单 GPU，副本占一块，通常可用多 GPU 执行[数据并行](../../training/model-parallelism/README.md#data-parallelism)。
- 装不下时，需要[张量并行](../../training/model-parallelism/README.md#tensor-parallelism)、[流水线并行](../../training/model-parallelism/README.md#pipeline-parallelism)或 [ZeRO 数据并行](../../training/model-parallelism/README.md#zero-data-parallelism)等分片。

副本数对应可用数据流数，也就是 DP。
- 单卡可容纳模型时，数据流数等于 GPU 数，DP=N_GPUS。
- 三维并行时，`DP=N_GPUs/(TP*PP)`；ZeRO 并行时，DP=ZeRO-DP。

全局批量为：

''',
'''

例如 8 块 GPU、MBS=4，使用 DP，则：

''',
'''

如果 TP=2、PP=2，每个副本占 4 块 GPU（`TP*PP`），总 GPU 数为 8，则：

''',
'''\n\n此时 GBS 为：\n\n''',
'''

使用[梯度累积](#gradient-accumulation)时，通常还定义累积多少步再同步，称为 GAS。若 GAS=4，每四步同步一次梯度，TP=1、PP=1、DP=8：

''',
'''

通常应让微批量尽可能大，使显存接近占满，但留有余量。

大模型剩余显存少，因此多装入一个样本也很重要。

序列长度、隐藏维度等最好是较大的 2 的倍数，例如 64、128，以获得最佳性能；但多数模型计算时会合并批量与序列维度，所以微批量自身的对齐通常影响很小，甚至没有影响。

因此，MBS=8 会 OOM、7 可以，就用 7，而不是退到 4。批量越大，单步处理的样本越多。
''']
assert len(parts)==len(codes)+1
with (R/'content'/p).open('a') as f:f.write('\n'+''.join(t+(codes[i] if i<len(codes) else '') for i,t in enumerate(parts)))
