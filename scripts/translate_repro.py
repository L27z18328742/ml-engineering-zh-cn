from translate_intro import interleave
interleave('training/reproducibility/README.md',[
'''# 可复现性

## 在依赖随机性的软件中实现确定性

调试时，始终为使用到的所有随机数生成器（RNG）设置固定种子，让每次重新运行都得到相同的数据和代码执行路径。

在 GPU 上，仅设置种子还不足以让结果逐位可复现：对于同一个操作，cuDNN 和 cuBLAS 可以在不同运行中自由选择不同的算法，而每种算法本身都可能是有效的。因此，真正确定性的重复运行，还要求它们始终选择相同且具有确定性的算法。下面的 `cudnn.benchmark`、`use_deterministic_algorithms()` 和 `CUBLAS_WORKSPACE_CONFIG` 设置就是为此服务的。

涉及的系统很多，很难全部覆盖。下面尝试涵盖最常见的确定性设置：

''',
'''

说明：

- 不需要的 RNG 种子设置可以注释掉，例如没有直接使用 `numpy` 时。不过，依赖的某个模块可能在你不知道的情况下内部使用它，因此通常最好全部保留，以免遗漏。

- `CUBLAS_WORKSPACE_CONFIG` 弥补了仅靠 `use_deterministic_algorithms(True)` 无法解决的问题。cuBLAS 是 `torch.mm`／`mv`／`bmm` 等基于 `matmul` 的操作背后的库。在 CUDA >= 10.2 上，它可能不具有确定性，因为同一句柄上的并行 CUDA stream 中运行的例程会动态选择内部工作区内存，因此即使种子相同，不同运行的结果仍可能不同。如果没有设置这个环境变量，`use_deterministic_algorithms(True)` 在遇到 `matmul` 操作时就会抛出 `RuntimeError`，这就是需要设置它的信号。**必须在创建 CUDA 上下文之前设置**，也就是在第一次 CUDA 调用之前；如果进程中的其他位置已经初始化 CUDA，之后再设置就会被静默忽略。因此，上面的 `enforce_reproducibility()` 在最开始、所有 `torch.cuda.*` 调用之前设置它。

  环境变量被静默忽略很容易漏察，因此上面的代码使用 [`torch.cuda.is_initialized()`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_initialized.html) 检查：如果执行 `enforce_reproducibility()` 时，`CUBLAS_WORKSPACE_CONFIG` 尚未设置，而之前某次调用已经触发 CUDA 延迟初始化，例如张量 `.cuda()`、`torch.cuda.set_device()`，或另一个库先访问了 CUDA，就直接抛出异常。注意，这只能发现经由 PyTorch 自己的延迟初始化路径发生的初始化。如果某个非 PyTorch 库先创建了原生 CUDA 上下文，`is_initialized()` 无法得知。因此，应尽可能在程序最早的位置调用 `enforce_reproducibility()`。

  NVIDIA 的 cuBLAS 文档明确认可两个用于确定性行为的值，两者都涉及显存与性能之间的取舍：较小的工作区使 cuBLAS 可选的算法更少，可能限制性能；较大的工作区则为选择更快算法留出余地，但需要额外显存。
  - `:16:8`：具有确定性，额外内存开销最小，但工作区较小可能限制性能。
  - `:4096:8`：具有确定性，需要额外约 24 MiB GPU 显存，但性能余地更大。两者中它更常用，上面的代码也采用它。

  不存在第三种完全无代价的选项。确定性在结构上就与“允许 cuBLAS 在并发 stream 中自由选择最快的算法和工作区布局”相冲突，因此将其固定下来必然牺牲一部分自由度。`:4096:8` 通常性能损失较小，因为工作区更大，排除的算法更少，但“较小”并不等于“零”。实际损失取决于工作负载和 `matmul` 的形状，可能微乎其微，也可能相当明显；一些报告显示，特定工作负载的吞吐量会下降约一半。

  通用格式是 `:[SIZE]:[COUNT]`，也可以连接多个 `SIZE:COUNT` 对，例如默认未设置时的行为等价于 `CUBLAS_WORKSPACE_CONFIG=:4096:2:16:8`。这种串联形式来自 cuBLAS 自己的内部工作区池配置，其中每一对最初描述一种不同的桶大小及数量，以高效满足大小不同的工作区请求。不过，PyTorch 只是将找到的每一对 `SIZE * COUNT` 相加，得到一个总字节数，再将这个固定大小的缓冲区交给 cuBLAS。因此，`:4096:2:16:8` 仅意味着总计 `2*4096 + 8*16` KiB。串联更多对，只会改变*相加得到多少 KiB*，不会改变确定性行为。实际使用时不需要自行串联：选用上面 NVIDIA 认可的两个单对值之一即可；也可以设置 `:0:0`，强制 cuBLAS 完全不使用工作区，但不推荐，因为这限制最多、速度最慢。

- `torch.use_deterministic_algorithms(True)` 让 cuDNN 卷积具有确定性，还覆盖 PyTorch 中数百种其他操作，例如 `index_add_`、`scatter_`、插值等。它不会静默地“尽力而为”，而是在遇到没有确定性实现的操作时立即抛出 `RuntimeError`，避免非确定性悄悄混入。

- `torch.backends.cudnn.benchmark = False` 是一个独立且互补的开关。它阻止 cuDNN 在首次遇到新的输入形状时，对多个卷积算法计时，并缓存最快的算法。这个基准测试步骤本身就会导致不同运行之间的差异：哪个算法在计时竞赛中获胜，可能取决于偶然的系统噪声，而与最终选中的算法自身是否具有确定性无关。

> **PyTorch 版本说明**：`torch.use_deterministic_algorithms()` 要求 **PyTorch >= 1.8**。对于 PyTorch < 1.8，可以退回使用范围更窄的 `torch.backends.cudnn.deterministic = True`。它只能让 cuDNN 卷积具有确定性，不会影响其他操作，例如 `matmul`、`index_add_` 和 `scatter_`。`CUBLAS_WORKSPACE_CONFIG` 与 PyTorch 版本无关，仍然适用。
> ''',
'''

参考资料：
- [NVIDIA cuBLAS 文档：可复现性](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility)：对 cuBLAS 非确定性和受支持的 `CUBLAS_WORKSPACE_CONFIG` 取值的权威说明。
- [`torch.use_deterministic_algorithms`](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html)：受影响操作的完整列表，以及 `warn_only` 的工作方式。
- [PyTorch CUDA 环境变量参考](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)：`CUBLAS_WORKSPACE_CONFIG` 的准确 `:[SIZE]:[COUNT]` 格式规范。
- [PyTorch 可复现性说明](https://docs.pytorch.org/docs/stable/notes/randomness.html)：结合种子、`cudnn.deterministic` 和 `cudnn.benchmark`，给出完整说明。

如果代码除了 CPU／CUDA，还使用其他加速器后端或框架，例如 NPU、XPU 或 TensorFlow，也要设置它们的 RNG 种子：

''',
'''

为了排查某个问题而反复运行同一份代码时，在程序开头使用以下调用设置指定种子：

''',
'''

如上所述，**这仅用于调试**：它会启用多种帮助实现确定性的 `torch` 标志，但可能降低速度，因此不适合直接用于生产。

生产环境中则可以改为调用：

''',
'''

也就是不显式传入种子。这样会随机选择一个种子，并将其写入日志！生产环境出问题时，就能重现问题发生时的 RNG 状态。而且这次没有性能惩罚，因为只有显式传入种子时，才会设置确定性选项（`cudnn.benchmark`／`use_deterministic_algorithms`／`CUBLAS_WORKSPACE_CONFIG`）。假设日志记录了：

''',
'''

那么只需把代码改为：

''',
'''

就能得到相同的 RNG 设置。

如前所述，一个系统还可能涉及许多其他 RNG。例如，想让 `DataLoader` 以同样顺序提供数据，也需要[设置它的种子](https://docs.pytorch.org/docs/stable/notes/randomness.html#dataloader)。

另请参阅调试章节中的[不同设备上的浮点运算差异](../../debug/pytorch.md#floating-point-math-discrepancies-on-different-devices)。

## 复现软件和系统环境

发现结果存在差异时，例如质量或吞吐量不同，这种方法会很有用。

其思路是记录启动训练或推理时环境中的关键组件，以便之后需要时能够精确重建当时的环境。

所使用的系统和组件种类繁多，不可能给出一种永远有效的方法。下面讨论一种可行方案，你可以据此适配自己的环境。

将下面这段 Bash 脚本添加到 Slurm 启动脚本，或你采用的其他训练启动流程中：

''',
'''

可以看到，这个方案用于 SLURM 环境，因此每次新训练都会导出该 SLURM 作业对应的环境。也可以将它适配到其他环境。

逐步说明如下：

1. 保存已加载的 `modules`。例如，在云集群／HPC 环境中，你可能通过它来加载 CUDA 和 cuDNN 库。

   如果不使用 `modules`，删除对应命令。

2. 导出 shell 环境变量。这可能很关键，因为在某些环境中，仅 `LD_PRELOAD` 或 `LD_LIBRARY_PATH` 一个环境变量，就可能对性能产生巨大影响。

3. 导出 Python 环境中安装的软件包及版本。这应该适用于 `conda`、`venv` 等任意 Python 虚拟环境，即使不使用虚拟环境也可以。

   如果使用 `uv` 而不是 `pip`，改用 `uv pip freeze`，速度会快得多。

4. 如果通过 `pip install -e .` 进行开发模式安装，除了 git SHA，它不了解安装来源的 Git 仓库的其他状态。问题在于，你很可能修改过本地文件，而 `pip freeze` 会遗漏这些改动。因此，这部分代码会通过 `site-packages/*.dist-info/direct_url.json` 找到没有直接安装到 conda 环境内的那些包，然后逐一检查；凡是存在未提交修改的，就将其 `git diff` 保存到 `$REPRO_DIR/<package>.diff`。

要保存 `apt` 软件包列表，加入：

''',
'''

如果使用 `conda`，另一个有用的工具是 [conda-env-compare.pl](https://github.com/stas00/conda-tools/blob/master/conda-env-compare.md)，它能找出两个 conda 环境的准确差异。

举个亲身经历的例子：我和同事在云集群上运行完全相同的代码，训练 TFLOPS 却相差很大；我们甚至是从同一个共享目录启动同一份 Slurm 脚本。起初用 [conda-env-compare.pl](https://github.com/stas00/conda-tools/blob/master/conda-env-compare.md) 比较 conda 环境，确实发现了一些差异。我安装了与她完全相同的软件包以对齐环境，但性能仍然相差巨大。后来比较 `printenv` 的输出，发现我设置了 `LD_PRELOAD`，她却没有。这产生了很大影响，因为那家云服务商要求把多个环境变量设置为自定义路径，才能充分发挥硬件性能。
'''])
