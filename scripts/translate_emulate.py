from translate_intro import interleave
interleave('training/emulate-multi-node.md',[
'''# 仅用单节点模拟多节点环境

目标是使用一台带有 2 块 GPU 的节点，模拟双节点环境，供测试使用。当然，还可以进一步扩展到[更大的配置](#larger-set-ups)。

这里使用 `deepspeed` 启动器，不必真正使用 DeepSpeed 的其他代码，只是它提供的高级功能更方便。只需先执行 `pip install deepspeed` 安装即可。

完整配置步骤如下。

1. 创建 `hostfile`：

''',
'''\n\n2. 为 SSH 客户端添加匹配的配置：\n\n''',
'''

如果端口不是 22，或主机名不是 `localhost`，请相应修改。

3. 本地环境很可能需要密码登录，因此要将自己的公钥添加到 `~/.ssh/authorized_keys`。

`deepspeed` 启动器明确使用无密码连接。例如，在 worker0 上会执行 `ssh -o PasswordAuthentication=no worker-0 hostname`，因此可以随时用下面的命令调试 SSH 配置：

''',
'''\n\n4. 创建测试脚本，确认两块 GPU 都被使用。\n\n''',
'''\n\n运行：\n\n''',
'''

如果 SSH 配置正常，可以同时运行 `nvidia-smi`，观察到 `torch.ones` 调用让两块 GPU 都分配了约 4 GiB 显存。

注意，脚本通过修改 `CUDA_VISIBLE_DEVICES`，让第二个进程使用 gpu1，但两个进程看到的都是 `local_rank==0`。

5. 最后，测试 NCCL 集体通信是否也能正常工作。

下面的脚本改编自 [torch-distributed-gpu-test.py](../debug/torch-distributed-gpu-test.py)，只增加了对 `os.environ["CUDA_VISIBLE_DEVICES"]` 的调整。

''',
'''\n\n运行：\n\n''',
'''

这样就完成了。

我们确认了 NCCL 集体通信可以工作，但这里走的是本地 NVLink／PCIe，而不是真实多节点中的 IB／以太网连接。因此，这是否足以满足测试需求，要取决于具体测试什么。

## 更大的配置

假设有 4 块 GPU，想模拟 2 个节点、每节点 2 块 GPU，只需把 `hostfile` 改成：

''',
'''\n\n并将 `CUDA_VISIBLE_DEVICES` 的调整代码改为：\n\n''',
'''

其他部分应保持相同。

## 自动化配置过程

如果希望自动处理任意拓扑形状，可以采用类似下面的方法：

''',
'''

## 用单块 GPU 模拟多块 GPU

下面讨论的需求与本文前面的需求相互独立，但也有关联，因此我认为值得分享一些经验。

NVIDIA A100 可以通过 [MIG](https://www.nvidia.com/en-us/technologies/multi-instance-gpu/) 将一块物理 GPU 模拟成最多 7 个 GPU 实例，但多年来，这些实例只能独立使用，例如不能在它们之间运行 DDP 或任何 NCCL 通信。我曾希望把自己的 A100 划成 7 个实例，再增加一块物理 GPU，凑成 8 块 GPU 用于开发，但没有成功。当时询问 NVIDIA 工程师，得到的答复是没有支持这种用法的计划。

现在情况变了：[NCCL 2.31（2026 年）增加了跨 MIG 实例通信器的实验性支持](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-mig-instances)。因此，一块物理 GPU 如今可以充当一个小型多 GPU 节点，在它的多个实例之间运行 DDP、FSDP、ZeRO 和任意 NCCL 集体通信。下面给出一个在 H200 上可运行的方案，以及这些模拟 GPU 之间实际通信速度的测量结果。

### 获取足够新的 NCCL

需要支持 CUDA 13.0 或更新版本的 NVIDIA 驱动，以及 NCCL 2.31 或更新版本。

截至 2026 年 8 月，最新的 PyTorch 2.13.0 捆绑的 NCCL 版本仍然较低。在出现捆绑 NCCL >= 2.31 的 PyTorch 构建之前，可以自行安装 NCCL，并优先加载它：

''',
'''

将它安装到独立目录，而不是当前环境中，是因为 PyTorch 固定了精确的 NCCL 版本，2.13.0 对应 `nvidia-nccl-cu13==2.29.7`。如果覆盖安装，会让 pip 的依赖解析与 torch 元数据发生冲突。`LD_PRELOAD` 则让动态链接器优先从新库解析 NCCL 符号，再考虑 torch wheel 内附带的版本。

为确认实际加载的版本，用 `NCCL_DEBUG=INFO` 运行一次程序，检查启动信息是否显示 `NCCL version 2.31.2+cuda13.3` 或更新版本。不要用 `torch.cuda.nccl.version()` 检查这一点，因为它报告的是构建 PyTorch 时使用的版本，而不是当前通过预加载实际运行的版本。

### MIG 所需的 NCCL 设置

''',
'''

NCCL 文档说不需要设置 `NCCL_MULTI_RANK_GPU_ENABLE`，因为每个 MIG 实例都是独立的 CUDA 设备。但在 H200 上使用 NCCL 2.31.2 时，没有这个设置，通信器就无法启动，因此还是设置上。NCCL 根据父 GPU 的 NVML 索引和 PCI bus id 来识别 rank 使用的设备，而不是根据 MIG 实例；于是，一块 GPU 上四个实例中的四个 rank，在它看来像是四个 rank 共用同一设备。不启用该标志，初始化就会报 `Multiple Ranks are using the same GPU/Partition`。

`NCCL_NVLS_ENABLE=0` 是防护措施，而不是必需项。在这种情况下，NCCL 默认就不会启用 NVLink SHARP；如果环境中没有其他地方设置这个变量，可以不管它。但许多集群镜像和 Pod 配置会导出 `NCCL_NVLS_ENABLE=1`。显式要求 NVLink SHARP 时，NCCL 会拒绝启动，并报 `NCCL_NVLS_ENABLE has been set to "1" and communicator has multiple ranks using the same NVML device`。设为 0 没有额外代价，也能避免已有环境变量影响该方案。也可以设为 `NCCL_NVLS_ENABLE=2`，在可用时自动使用 NVLS。

文档还要求设置 `NCCL_MNNVL_ENABLE=0`。在没有 NVLink fabric 的节点上，这个标志没有影响：相同基准测试在未设置、设置为 `0`、设置为 `1` 时分别得到 142.28、142.42 和 142.23 GBps。因此，它只对 GB200 NVL72 这类接入 fabric 的系统有意义。

### 划分 GPU

先列出 GPU 支持的配置，启用 MIG，再创建实例。H200 最多可创建 7 个实例。这里在 GPU 4 上创建 4 个 35 GB 实例，这样就可以在单块 GPU 上测试例如 SP=2、EP=2 的模型并行。

''',
'''\n\n要撤销 MIG、恢复使用完整 GPU：\n\n''',
'''

### 查找 MIG UUID

必须通过 `CUDA_VISIBLE_DEVICES` 为每个 rank 分配恰好一个实例。MIG 实例使用 UUID 寻址，而不是索引。`nvidia-smi -L` 会在父 GPU 下面缩进列出它们：

''',
'''

将节点上每个可供一个 rank 使用的设备汇总到一个逗号分隔的变量中：开启 MIG 的 GPU 提供其各个实例，未开启 MIG 的 GPU 则以完整设备加入。

''',
'''

### 启动各个 rank

`torchrun` 会为每个进程设置 `LOCAL_RANK`，据此就能为它们分配各自的实例：

''',
'''

必须将 `LOCAL_RANK` 重置为 `0`，参见[注意事项](#caveats)。

`all_reduce`、`all_gather`、`reduce_scatter`、`broadcast` 和 DDP 训练步，都可以在这样的 4 个实例之间运行。

### 模拟 GPU 之间的通信有多快？

只有集体通信不至于慢得难以使用，这种模拟才有价值。因此，我们在一块 H200 的 MIG 实例上运行 [torch-dist-bench.py](../network/benchmarks/torch-dist-bench.py)，与相同数量的完整 H200 对比：4 个实例对比 4 块真实 GPU，2 个实例对比 2 块真实 GPU。

下面使用的配置都来自一块 H200 141GB：

| MIG 配置 | 计算切片 | 显存 | 每块 GPU 的实例数 |
| :---------- | -------------: | -----: | ----------------: |
| `1g.18gb` | 7 份中的 1 份 | 18GB | 7 |
| `1g.35gb` | 7 份中的 1 份 | 35GB | 4 |
| `2g.35gb` | 7 份中的 2 份 | 35GB | 3 |
| `3g.71gb` | 7 份中的 3 份 | 71GB | 2 |

下面每张结果表先列出作为对照的完整 GPU，再按 `busbw` 降序列出 MIG 结果。

4 个 rank，512 MiB 负载：

| 4 个 rank 的运行设备 | busbw | 相对完整 GPU |
| :---------------------- | --------: | ------------: |
| 4 块完整 H200（NVLink） | 341.1GBps | 100% |
| 一块 GPU 的 4 个 `1g.35gb` 实例 | 273.8GBps | 80% |
| 一块 GPU 的 4 个 `1g.18gb` 实例 | 142.3GBps | 42% |

4 个 rank，2 GiB 负载：

| 4 个 rank 的运行设备 | busbw | 相对完整 GPU |
| :---------------------- | --------: | ------------: |
| 4 块完整 H200（NVLink） | 354.7GBps | 100% |
| 一块 GPU 的 4 个 `1g.35gb` 实例 | 281.3GBps | 79% |
| 一块 GPU 的 4 个 `1g.18gb` 实例 | 144.7GBps | 41% |

2 个 rank，512 MiB 负载：

| 2 个 rank 的运行设备 | busbw | 相对完整 GPU |
| :---------------------- | --------: | ------------: |
| 2 块完整 H200（NVLink） | 308.2GBps | 100% |
| 一块 GPU 的 2 个 `3g.71gb` 实例 | 413.8GBps | 134% |
| 一块 GPU 的 2 个 `2g.35gb` 实例 | 245.3GBps | 80% |
| 一块 GPU 的 2 个 `1g.35gb` 实例 | 230.9GBps | 75% |
| 一块 GPU 的 2 个 `1g.18gb` 实例 | 122.0GBps | 40% |

2 个 rank，2 GiB 负载：

| 2 个 rank 的运行设备 | busbw | 相对完整 GPU |
| :---------------------- | --------: | ------------: |
| 2 块完整 H200（NVLink） | 331.7GBps | 100% |
| 一块 GPU 的 2 个 `3g.71gb` 实例 | 429.5GBps | 129% |
| 一块 GPU 的 2 个 `2g.35gb` 实例 | 250.4GBps | 75% |
| 一块 GPU 的 2 个 `1g.35gb` 实例 | 235.0GBps | 71% |
| 一块 GPU 的 2 个 `1g.18gb` 实例 | 123.6GBps | 37% |

上面的配置表中，计算切片数就是配置名称中的数字，例如 `2g.35gb` 获得 GPU 的 7 个计算切片中的 2 个。显存容量则反映实例拥有 8 个显存切片中的多少个：18 GB 对应 1 个，35 GB 对应 2 个，71 GB 对应 4 个。

测试环境：H200 141GB SXM，`torch==2.9.1+cu130`，按上述方法预加载 NCCL 2.31.2，CUDA 驱动 13.0。

最大的意外是，不同配置对结果的影响如此之大。MIG 实例间的流量从不离开 GPU，NCCL 日志中所有通道都是 `P2P/CUMEM`，整个运行过程中没有 socket 或 PCIe 回退。因此，瓶颈是实例拥有的显存带宽，由显存切片决定，几乎不取决于 SM 数量。两张 512 MiB 表清楚地隔离了这一点：4-rank 表中，从 `1g.18gb` 改为 `1g.35gb`，计算资源仍为一个切片，显存翻倍，带宽也几乎翻倍（142 → 274 GBps）；2-rank 表中，从 `1g.35gb` 改为 `2g.35gb`，显存不变、计算资源翻倍，带宽却只提高了 6%（231 → 245 GBps）。

因此，4 个 `1g.35gb` 实例能达到 4 块真实 H200 约 79% 的互联带宽，而 rank 数相同、切片更小的 4 个 `1g.18gb` 实例只有 41%。应选择在满足所需 rank 数的前提下，每个实例显存最大的配置。

如果只有 2 个 rank，`3g.71gb` 实例实际上比 2 块完整 GPU *更快*：在 2-rank、2 GiB 表中是 429 对 332 GBps，因为单芯片 HBM 内部复制比经过 NVLink 跳转更快。

### 将实例分散到两块 GPU 上

在 NVIDIA 解决奇怪的[每块 GPU 最多 7 个实例](#only-7-instances-per-gpu)限制之前，如果想运行 [3D 并行](model-parallelism/README.md#dppptp)，至少需要 8 块 GPU 或 8 个实例，因此目前必须用两块 GPU 才能凑够 8 个。

上面的所有模拟都位于同一块物理 GPU 内。把两块 GPU 各划分为 4 个实例，可以得到 8 个 rank。配置与基准测试如下：

''',
'''

8 个 rank，512 MiB 负载：

| 8 个 rank 的运行设备 | busbw | 相对完整 GPU |
| :----------------------------- | --------: | ------------: |
| 8 块完整 H200（NVLink） | 450.2GBps | 100% |
| 两块 GPU，各 4 个 `1g.35gb` 实例 | 26.8GBps | 6% |

8 个 rank，2 GiB 负载：

| 8 个 rank 的运行设备 | busbw | 相对完整 GPU |
| :----------------------------- | --------: | ------------: |
| 8 块完整 H200（NVLink） | 465.9GBps | 100% |
| 两块 GPU，各 4 个 `1g.35gb` 实例 | 25.6GBps | 5% |

性能出现了断崖式下降，而且不是消息太小造成的：2 GiB 负载也没有优于 512 MiB。NCCL 自己的日志解释了原因：ring 通道中，24 个跳转走 `via P2P/CUMEM`，6 个走 `via SHM/direct`。留在同一 GPU 内的跳转仍走高速路径，但跨到另一块 GPU 的跳转回退到了主机共享内存，因为 CUDA 驱动不支持 MIG 实例之间的 NVLink P2P；这也是这里无法使用 NVLink SHARP 的原因。ring 的速度受最慢链路限制，因此那几个经过主机内存的跳转决定了全部 8 个 rank 的速度。

### 混合 MIG 实例与完整 GPU

不必把每块 GPU 都划分成实例。由于一块 GPU 最多提供 7 个实例，组成 8-rank 环境成本最低的方法，是把一块 H200 或其他高端 GPU 划成最多的 7 个实例，另一块 GPU 保持完整。前面的 `MIG_UUIDS` 命令已经能为这种混合方式生成正确列表：划分过的 GPU 提供 7 个实例，完整 GPU 则直接加入。

''',
'''

8 个 rank，512 MiB 负载：

| 8 个 rank 的运行设备 | busbw | 相对完整 GPU |
| :------------------------------------- | --------: | ------------: |
| 8 块完整 H200（NVLink） | 450.2GBps | 100% |
| 一块 GPU 的 7 个 `1g.18gb` 实例，加一块完整 H200 | 28.5GBps | 6% |

8 个 rank，2 GiB 负载：

| 8 个 rank 的运行设备 | busbw | 相对完整 GPU |
| :------------------------------------- | --------: | ------------: |
| 8 块完整 H200（NVLink） | 465.9GBps | 100% |
| 一块 GPU 的 7 个 `1g.18gb` 实例，加一块完整 H200 | 27.5GBps | 6% |

完整 GPU 也无法挽救性能：划分过的 GPU 内有 24 个跳转走 `via P2P/CUMEM`，到完整 GPU 的 6 个跳转走 `via SHM/direct`，ring 仍受主机内存跳转速度限制。由此可以得到一个实用结论：第二块设备自身的带宽并不重要。第八个 rank 放在完整 H200 上得到 28.5 GBps，与放在第二块已划分 H200 的 MIG 实例上得到的 26.8 GBps 相比，并没有实质改善。因此，如果只是为了突破每块 GPU 7 个实例的上限，第二块设备可以是 CUDA 能使用的任意 GPU；它以完整设备加入，不需要支持 MIG。一块 H200 加一块普通 GPU，就能组成 8-rank 环境。

实践原则是：尽量让模拟的多 GPU 作业留在同一块物理 GPU 内。与一块 GPU 上的 4 个实例相比，跨到第二块 GPU 会慢约 10 倍；512 MiB 时是 26.8 对 273.8 GBps，只剩 8 块真实 GPU 性能的约 5%。不过，如果用于开发，模型和通信负载都很小，这可能并不重要。

请记住，它模拟的是多 GPU 作业的*拓扑*，不是性能。每个 rank 只得到一部分 SM 和显存，因此这是开发和测试工具，不能用来测量真实吞吐量。

### 注意事项

#### 每个 rank 的 `LOCAL_RANK` 都是 `0`

一旦 `CUDA_VISIBLE_DEVICES` 指定了单个实例，进程就只能看到一个设备，CUDA 将其编号为 `0`。如果代码直接使用 torchrun 分配的值调用 `torch.cuda.set_device(local_rank)`，就会报 `invalid device ordinal`。无法绕过这一点：即使多个 MIG 设备可见，CUDA 也只会为每个进程枚举一个计算实例。rank 分布在两块已划分 GPU 上时也是如此；rank 落在哪块物理 GPU 上，由 UUID 决定，而不是设备序号。因此，每个 rank 都显示 `cuda:0`，实际却位于不同硬件资源上。

于是每个 rank 都会报告 `LOCAL_RANK=0`，这里需要准确区分其影响。

`torch.distributed` 不受影响。`dist.get_rank()`、`dist.get_world_size()` 和进程组依据 `RANK` 和 `WORLD_SIZE` 构建，启动包装脚本不会修改它们。常见的设备选择写法也可以保持不变：

''',
'''

每个进程请求的都是设备 0，却获得不同切片，因为各 rank 的 `CUDA_VISIBLE_DEVICES` 不同。

会出问题的是将 `LOCAL_RANK` 当成身份标识，而不是设备序号的代码。常见例子是判断节点主进程：

''',
'''

在 MIG 下，这个条件对每个进程都成立，因此所有 rank 都会下载数据集、写同一个日志文件，或竞争填充同一个缓存目录。

单节点模拟时，可以改用整个作业内仍然唯一的 `RANK`：

''',
'''

如果代码还必须不经修改地用于真实多节点运行，就使用另一个变量保存原始槽位编号。启动包装脚本在覆盖前先保存它：

''',
'''\n\n判断条件则读取当前存在的变量：\n\n''',
'''

#### 每块 GPU 只有 7 个实例

MIG 沿芯片现有的边界划分资源。计算切片来自 GPC（Graphics Processing Cluster，图形处理簇），这个顶层模块包含一组 SM 以及自己的工作分发机制；显存切片则来自与 HBM 堆栈连接的内存系统。根据 NVIDIA 的资料，Ampere、Hopper、Blackwell，以及最近的 Rubin，完整 GPU 实现都恰好有 8 个 GPC，虽然出货 SKU 可能只启用其中一部分。

MIG 最多支持 7 个实例而不是 8 个，是因为第八个计算单元预留给实例管理。只要启用 MIG 模式，无论随后创建 7 个实例还是只创建 1 个，它都会被保留。显存系统没有这样的占用，因此 8 个显存切片仍全部可用。MIG 文档明确给出了由此产生的比例：每个显存切片占总显存的 1/8，每个 SM 切片占 SM 的 1/7。

上面的配置表体现了这种不对称性。7 个 `1g` 实例消耗全部 7 个计算切片，却只使用 8 个显存切片中的 7 个，因此完全划分后的 GPU 会有一个切片的 HBM 无法利用；对 141 GB 的 H200 而言，就是 18 GB。这也是 `4g` 配置只能创建一个实例的原因：第二个实例需要总计 8 个计算切片，但只有 7 个可用。

#### `torch.cuda.device_count()` 与 `LOCAL_WORLD_SIZE` 不一致

一个进程只能看到自己的实例，因此每个 rank 调用 `torch.cuda.device_count()` 都返回 1。在使用完整 GPU 时，这个调用的结果恰好等于节点上的 rank 数，因此不少代码依赖它来设置分片数、构建设备遍历列表，或检查请求的 world 是否能容纳在节点中。这里应改为从环境变量读取数量：

''',
'''

torchrun 会在每个 worker 中同时设置 `LOCAL_WORLD_SIZE`、`RANK`、`LOCAL_RANK` 和 `WORLD_SIZE`。在 MIG 下，`LOCAL_WORLD_SIZE` 仍然报告完整的本地 rank 数量。
'''])
