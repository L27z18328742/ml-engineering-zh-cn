# 容错

无论机器学习训练硬件是自有还是按小时租用，在发展不断加速的机器学习领域，及时完成训练都很重要。如果你睡觉时某块 GPU 故障，或检查点存储空间耗尽，导致训练崩溃，醒来时就会发现已经损失了许多小时的训练时间。

机器学习硬件成本极高，因此很难像 Web 服务那样提供冗余故障切换。不过，只需几个简单方法，就能让训练具备容错能力。

大多数严肃的训练任务都在 SLURM 环境中运行，因此本章会经常提到它，但大部分经验也适用于其他训练环境。

## 始终准备多于实际需求的节点

GPU 设备的现实是，它们容易发生故障。有时只是过热关机，随后能恢复；有时则真的损坏，必须更换。

连续使用同一批节点几周或几个月后，随着故障设备逐步被替换，情况通常会改善。但如果拿到一批新 GPU，尤其是某项技术刚推出时的早期产品，就要预期其中相当一部分可能故障。

因此，如果训练需要 64 个节点，要确保还有几个备用节点，并提前了解：当现有备用节点不够时，能够多快替换故障节点。

很难预测准确的冗余比例，但 5–10% 应该是合理的。按时完成训练的压力越大，安全余量就应越高。

准备好备用节点后，要验证 SLURM 会自动将问题节点移出可用池，并自动用正常节点替换故障节点。

使用非 SLURM 调度器时，也要验证它能否无人值守地替换故障节点。

另外至少还需要一个节点，用于运行本章后面介绍的预防性 watchdog，也可能用于卸载检查点和执行清理任务。

## 原地重建进程组

本章其余部分通过退出程序，让排队中的下一个作业获得新资源来处理 rank 故障。从 PyTorch 2.14 开始，可以原地重建通信器，让存活 rank 保留已经就绪的进程状态。截至 2.14，这条路径仍是实验性的：在 `init_process_group(..., enable_reconfigure=True)` 中启用，相关调用 `dist._get_reconfigure_handle`、`dist._reconfigure` 带下划线，文档也说明可能变化。在它们稳定之前，仍应使用备用节点和作业数组作为恢复路径。

## 排队提交多个训练作业

接下来关键的一步，是确保训练崩溃后，已经有新作业排队接替它。

因此，启动训练时，不要只用：

```bash
sbatch train.slurm
```

而应替换为：

```bash
sbatch --array=1-10%1 train.slurm
```

这会告诉 SLURM 预订一个包含 10 个作业的数组。某个作业无论正常结束还是崩溃，都会立即调度下一个。

补充说明：`--array=1-10%1` 中的 `%1` 表示串行启动作业数组，一次只运行一个作业。

如果已经启动训练，却没有这样设置，也可以使用 `--dependency` 参数补救，不必终止当前作业：

```bash
sbatch --array=1-10%1 --dependency=CURRENTLY_RUNNING_JOB_ID train.slurm
```

例如，当前作业如下：

```bash
$ squeue -u `whoami` -o "%.10i %9P %20j %.8T %.10M %.8l %.6D %.20S %R"
     JOBID PARTITION NAME             STATE       TIME   TIME_LIM    NODES  START_TIME     NODELIST(REASON)
       87    prod    my-training-10b  RUNNING 2-15:52:19 1-16:00:00   64    2023-10-07T01:26:28 node-[1-64]
```

可以看到当前 `JOBID=87`，因此可以执行：

```bash
sbatch --array=1-10%1 --dependency=87 train.slurm
```

随后状态会变成：

```bash
$ squeue -u `whoami` -o "%.10i %9P %20j %.8T %.10M %.8l %.6D %.20S %R"
     JOBID PARTITION NAME             STATE       TIME   TIME_LIM    NODES  START_TIME     NODELIST(REASON)
       87    prod    my-training-10b  RUNNING 2-15:52:19 1-16:00:00   64    2023-10-07T01:26:28 node-[1-64]
 88_[1-10%1]   prod    my-training-10b  PENDING       0:00 1-16:00:00   64                    N/A (Dependency)
```

可以看到，新增的 10 个作业组成数组 `88_[1-10%1]`，会在当前作业 `87` 完成或失败后启动。

当然，如果导致崩溃的条件仍然存在，后续作业也会失败。例如存储已满，无论重启多少次都无法继续训练。后面会讨论如何避免这种情况。

不过，训练崩溃的主要原因是 GPU 故障，因此，只要确保自动移除故障节点，并让新作业使用一组新节点启动，就能顺利从崩溃中恢复。

在 SLURM 术语中，被移除的节点会被标记为 `drained`。下面是一个假想集群：

```bash
$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
prod*       up   infinite       4  drain node-[0-3]
prod*       up   infinite      47  alloc node-[4-51]
prod*       up   infinite      23   idle node-[52-73]
```

这里有 47 个节点正在使用（`alloc`），23 个可用（`idle`），4 个不可用（`drained`）。

系统管理员应定期检查 drained 节点，维修或更换后，将状态改回 `idle`，让它们重新可用。

另一种方法是通过 `--dependency` 将作业串联，参见[这里](../../orchestration/slurm/users.md#request-allocation-via-dependency)。两种方法也可以结合。

如何判断作业数组或串联作业何时不该再恢复？通常训练循环知道任务完成后，会立即退出。也可以增加[停止开关](#kill-switch)等功能，更方便地阻止作业数组继续运行。

## 优先选择固定加速器分配，而不是动态分配

刚拿到一组加速器节点时，尤其是刚发布的新型号，往往会有许多加速器故障，让 LLM 训练困难重重。新加速器早期故障率可能高达 10%，后续阶段也可能仍然偏高。记住，使用 8 个加速器时，从训练程序的角度看，一个故障往往就相当于 8 个都不能使用。

固定分配节点时，几个月后坏设备会逐渐被淘汰，故障应该很少再发生。仍然可能出现，但会变成偶发事件。

确保服务商在故障后提供新加速器，而不是让原设备降温后又交还给你。例如，可以跟踪 [NVIDIA GPU UUID](../../compute/accelerator/nvidia/debug.md#how-to-detect-if-you-get-the-same-broken-node-again-and-again)。这些瞬时故障在高负载下很可能复发，所以需要真正更换硬件。

动态分配时，即使新型号已发布一年，也要预期会拿到许多故障加速器，因为你可能接手其他用户退回的节点。有些云服务商确实更认真地更换坏硬件，但问题在于，许多加速器不会立即彻底失效。用户释放故障节点后，技术人员简单测试可能看不出问题。如果用户没有报告故障，而云服务商又不在交给下一位用户前重新检查，拿到坏节点的概率就会很高。

## 频繁保存检查点

训练作业失败时，可能损失许多小时的计算。频繁保存检查点可以缓解这个问题，恢复训练时从最近保存的位置继续。如果上次保存后过了 12 小时才故障，就必须重做这 12 小时的训练。使用数百块 GPU 时，这个代价非常高。

理论上可以每 10 分钟保存一次，让最多损失的训练时间只有 10 分钟。但大模型保存不够快，如果保存本身成为瓶颈，也会显著推迟训练完成，反而得不偿失。

根据检查点保存方式和存储分区的 I/O 速度，保存大模型可能需要几十秒到几分钟。因此，最佳保存频率位于两者之间。

计算很简单：测量一次保存耗时，乘以计划保存次数，看看它会给总训练时间增加多少延迟。

案例：训练 BLOOM-176B 时，我们使用非常快的 NVMe 上的 GPFS 文件系统，384 个进程并发写入一个 2.3 TB 检查点，只需 40 秒。大约每 3 小时保存一次，总训练约 3 个月，因此保存约 720 次（`90 days * 24h / 3h`）。单是保存检查点就额外花了 8 小时（`720 times * 40 secs / 3600 secs`），约占总训练时间的 0.37%（`8h / (90 days * 24 hours)`）。如果 I/O 慢 5 倍，这在不购买高价 I/O 的云环境中并不少见，就会占约 2% 的训练时间，相当可观。

补充说明：如果本地存储不大，必须把检查点卸载到云端，应将最近两个检查点留在本地，以便快速恢复。保留两个而不是一个，是因为若在保存期间崩溃，最新检查点可能损坏或未保存完整。

保存检查点虽然有额外开销，但非常有用。发生发散时可以回滚很多步，也方便分析各种事件。如今不少训练已经从训练过程中的单一损失评估，转向针对每个检查点、在多个基准上的完整数据集评估；前者提供的信号有限，后者可以在额外节点上运行，不必因为训练内评估而拖慢训练。

## 基于多副本的容错

处理加速器故障还有一种不依赖保存检查点的方法，但要求训练至少使用两个模型副本。

请先阅读各种[模型并行](../model-parallelism)技术，以便理解下面内容。

- 采用三维模型并行的某种组合，即 TP、PP、DP 中的一种或多种时，副本数等于 DP 并行度。
- 使用混合 ZeRO-DP 时，副本数等于混合副本的数量。

例如，训练配置为 TP=4、PP=2、DP=2，那么有两个副本，分别使用 `node0` 和 `node1` 的 8 块 GPU（`4*2=8`）；实际就是每个副本占用一个完整的 8-GPU 节点。

另外准备一个有 8 块 GPU 的备用节点 `node2`，平时空闲，但随时可投入使用。

假设训练中 `node0.gpu0` 故障。由于第二个副本的数据完好，可以切换到备用 8-GPU 节点，通过 RDMA 从第二个副本的 GPU 复制数据，然后从中断位置继续训练。这只是极简说明；根据故障发生在迭代循环的哪个阶段，恢复还涉及很多细节，也就是说，需要实现复杂算法。

当然，大规模训练中，通常会有约一百个活跃节点和少量备用节点。

这种方法优于文件系统检查点，因为最多损失一次迭代，而基于文件系统恢复可能损失数百次迭代。

最接近这一思路的开源库是 Meta／PyTorch 的 [`torchft`](https://github.com/meta-pytorch/torchft)。它将每个训练步视为副本组之间的故障边界，通过 lighthouse／quorum 服务协调健康状态，并能从正常对等副本实时传输状态，修复故障组，而不必回滚到文件系统检查点。它开箱支持容错 DDP 和 HSDP；[`torchtitan`](https://github.com/pytorch/torchtitan) 将其接入了完整 HSDP 训练循环，参见其 [`torchft` 实验文档](https://github.com/pytorch/torchtitan/tree/main/torchtitan/experiments/torchft)，以及 [PyTorch 关于在模拟故障下无检查点训练的博客](https://pytorch.org/blog/fault-tolerant-llama-training-with-2000-synthetic-failures-every-15-seconds-and-no-checkpoints-on-crusoe-l40s/)。故障域是整个副本组，而不是单块 GPU；修复通过 torchft 的检查点传输机制进行对等恢复，不是上面设想的 RDMA 复制到专用备用节点。目标相同——最多损失一步、继续训练——但实现机制不同。

相关研究系统从其他角度解决同一问题。[`Oobleck`](https://github.com/SymbioticLab/Oobleck)（SOSP'23）维持逻辑等价的流水线副本，故障后重新实例化预先计算的流水线模板，也无需完整检查点重启。更早的抢占式实例研究包括 [Bamboo](https://www.usenix.org/system/files/nsdi23-thorpe.pdf)（将冗余计算安排到流水线气泡中，NSDI'23）和 [Varuna](https://github.com/microsoft/varuna)（弹性 PP+DP 重配置）。这些代码库目前大多不再活跃，主要适合作为设计参考。商业基础设施中，也有无需修改训练脚本、就能把故障 GPU 迁移到备用设备的方案；它采用单 GPU 故障域，并工作在框架之外，属于另一种取舍，而不是又一个 torchft 复制品。

## 停止开关

在许多 SLURM 环境中，用户没有 `sudo` 权限。某位用户启动训练后去睡觉，其他人发现问题时，往往无法轻易停止并重启他的训练。

BLOOM-176B 训练时就是如此，因此我们实现了停止开关。机制很简单：训练循环每次开始新迭代前，检查一个指定文件是否出现。如果出现，就保存检查点并退出，让原启动者之外的用户也能修改配置并重新启动。我们还在 `main` 最开始加了一次检查。如果睡觉的用户排了很长的作业数组，后续作业就会在启动时迅速退出，从而快速“耗尽”它们。

[这里](../../orchestration/slurm/users.md#overcoming-the-lack-of-group-slurm-job-ownership)也讨论了这个问题。

这有助于减少浪费的训练时间。

## 保存开关

提到停止开关，也值得介绍它的近亲：保存开关。训练循环发现保存开关文件出现时，只保存检查点，继续训练，并自动删除该文件，避免之后每次迭代都保存一次。

对监看训练曲线的人来说，这很有用。发现损失或其他指标出现有趣、关键的状态时，可以立即请求训练程序保存当前检查点，以后随时重现这一状态。

主要用途是观察损失尖峰和发散。

## 预防

避免损失训练时间最简单的方法，是预防某些问题发生。除了确保散热足够，我们无法阻止 GPU 故障，但完全可以保证未来几天训练还有足够磁盘空间。通常通过定时运行 watchdog，监控资源，并在潜在问题发生很久之前提醒操作人员。

### 定时 watchdog

讨论具体 watchdog 前，必须先具备运行定时任务的机制。Unix 中通常由 [`crontab`](https://en.wikipedia.org/wiki/Cron) 实现。

例如，每小时运行一次 `~/bin/watch-fs.sh`：

```
0 * * * * ~/bin/watch-fs.sh
```

上面的链接解释了如何配置其他运行频率。

执行 `crontab -e` 设置任务，执行 `crontab -l` 查看已计划的任务。

这里不展开太多细节，是因为许多 SLURM 环境不允许使用 `crontab`，需要用其他方式调度任务。

[模拟 Crontab](../../orchestration/slurm/users.md#crontab-emulation)一节介绍了如何用 SLURM 实现类似功能，也可参见[自我延续的 SLURM 作业](../../orchestration/slurm/users.md#self-perpetuating-slurm-jobs)。

### 通知机制

接下来需要一种或多种通知方式。

最简单的是邮件告警。必须先确保能从 SLURM 作业发送邮件；如果环境没有提供，可以请系统管理员添加，也可以尝试使用外部 SMTP 服务商。

除了邮件，也可以设置短信通知；如果使用 Slack，则可以向指定频道发送通知。

掌握定时调度，并让通知机制正常工作后，再来看关键的 watchdog。

### 检查作业是否运行的 watchdog

最直观的是检查训练 SLURM 作业是否正在运行，或是否还有作业等待运行。

[slurm-status.py](slurm-status.py) 是 BLOOM-176B 训练中使用的例子。如果发现作业既没有运行，也没有排队，它会发送邮件，并把检查结果写入主训练日志。我们使用[模拟 Crontab](../../orchestration/slurm/users.md#crontab-emulation)，因此只需将 [slurm-status.slurm](slurm-status.slurm) 放入 `cron/cron.hourly/`，之前启动的 SLURM 调度器就会大约每小时执行一次检查。

该 SLURM 作业的关键部分是：

```bash
tools/slurm-status.py --job-name $WATCH_SLURM_NAME 2>&1 | tee -a $MAIN_LOG_FILE
```

它指定要监控的作业名称，同时将结果写入日志。例如，运行：

```
tools/slurm-status.py --job-name my-training-10b
```

而当前状态报告如下：

```bash
$ squeue -u `whoami` -o "%.10i %9P %20j %.8T %.10M %.8l %.6D %.20S %R"
  JOBID    PARTITION NAME             STATE       TIME   TIME_LIM    NODES  START_TIME     NODELIST(REASON)
    87     prod      my-training-10b  RUNNING 2-15:52:19 1-16:00:00  64    2023-10-07T01:26:28 node-[1-64]
```

那么一切正常。如果列表里没有 `my-training-10b`，就会发送告警。

只需少量修改路径和邮件地址，就能适配自己的需求。如果不是你启动的作业，就把 `whoami` 替换为实际启动用户的名称；只有自己启动作业时，`whoami` 才适用。

### 检查作业是否挂起的 watchdog

如果应用使用 `torch.distributed` 等机制，在集体通信中挂起，最终会超时并抛异常，然后重启训练，也可以发送重启告警。

但如果挂起发生在没有超时的其他系统调用中，例如磁盘读取，应用可能卡住数小时而无人知晓。

大多数应用会定期记录日志，例如每几分钟记录最近 N 步的训练统计。因此，可以检查日志是否在预期时间内更新；没有更新就告警。可以自己写，也可以使用 [io-watchdog](https://github.com/grondo/io-watchdog)。它从 2019 年以来没有提交，应视为功能可用但无人维护，而非持续开发中的项目。

### 磁盘空间不足告警

下一个大问题是磁盘空间耗尽。检查点很大、保存频繁，又没有卸载到其他地方，很容易快速耗尽空间。团队通常还共享集群，同事也可能迅速占用大量空间。理想情况下应有专用于训练的存储分区，但往往难以做到。无论如何，都必须知道何时空间不足、需要清理。

告警阈值不能太早，例如使用率才到 50% 就告警，用户会开始忽略它们。但百分比也未必适用：非常大的共享存储即使只剩 5%，也可能还有很多 TB；小分区即使剩 25%，也可能只有几 TB。因此，需要知道检查点保存频率、每天消耗多少 TB，以及当前可用空间。

案例：BLOOM 训练每 3 小时保存一个 2.3 TB 检查点，因此每天消耗 18.4 TB！

此外，往往存在多个分区：高 I/O 性能分区用于检查点，较慢分区存放代码和库，还可能使用其他分区。凡是其可用性会影响训练的，都需要监控。

还有一个陷阱：分布式文件系统未必能可靠提供所购买容量的 100%。有些最多只能可靠使用约 80%。它们底层的物理磁盘会按计划或触发条件重新均衡，但某块盘可能先达到 100%，导致写入失败、训练进程崩溃，而 `df` 显示整个分区才使用 80%。BLOOM-176B 没遇到这个问题，IDEFICS-80B 却遇到了：在那里，80% 就是新的 100%。如何得知存在这种问题？通常会在训练准备期间发现。

还不止如此，inode 可用量也是问题，有些分区的 inode 配额不高。Python 软件包通常包含数百到数千个小文件，总容量不大，但虚拟环境中累计可能有数万个文件。于是，明明还剩数 TB 空间，却因 inode 耗尽导致训练崩溃。

最后，许多分布式分区并不实时更新用量，可能要等 30 分钟。

补充说明：使用 `df -ih` 查看 inode 配额和使用量。

补充说明：有些文件系统内部压缩数据，报告的用量可能低于复制到其他地方后的实际占用，容易令人困惑。

这里是 BLOOM-176B 使用的 [fs-watchdog.py](./fs-watchdog.py)，任一存储条件不满足阈值时会发送邮件，对应的启动脚本是 [fs-watchdog.slurm](./fs-watchdog.slurm)。

阅读代码可以看到，每个分区同时监控磁盘容量和 inode。有些分区用 HPC 提供的专用配额工具获取即时统计，但这些工具不适用于所有分区，所以其他地方回退到 `df`，甚至更慢的 `du`。据此应很容易适配自己的环境。

### 处理缓慢的内存泄漏

有些程序存在很小的内存泄漏，极难调试。不要把它与 MMAP 混淆：MMAP 利用 CPU 内存快速读取数据，占用看似随时间增长，但这些内存在需要时会被释放，并非真正泄漏。可阅读[《深入调查 MMAP 为什么没有泄漏内存》](https://stasosphere.com/entrepreneur-being/301-mmap-memory-leak-investigation/)。

理想情况下当然应分析软件、修复泄漏，但问题可能来自第三方包，也可能很难诊断，而时间又不充裕。

GPU 显存还可能碎片化：随着运行时间增长，越来越多微小空闲片段累积，GPU 看似有不少可用显存，但申请大张量却会 OOM，例如：

```
RuntimeError: CUDA out of memory. Tried to allocate 304.00 MiB (GPU 0; 8.00 GiB total capacity;
142.76 MiB already allocated; 6.32 GiB free; 158.00 MiB reserved in total by PyTorch)
```

这个例子中，既然还有 6.32 GiB 可用，为何连 304 MiB 都无法分配？

IDEFICS-80B 训练期间，我们遇到很小的 CPU 内存泄漏，往往几天才耗尽内存。团队采用的一种方法，是在训练循环中安装 watchdog，监测内存用量，达到阈值就主动退出。下一个训练作业恢复时，CPU 内存已经全部回收。

补充说明：机器学习训练的现实是，有限资源下不可能解决所有问题。有时可靠的变通方法能让训练更早完成，而停下一切查根因，可能耽误数周。例如，BLOOM-176B 使用 `CUDA_LAUNCH_BLOCKING=1` 训练，因为不设置就会挂起。多次诊断失败后，我们无法继续等待，只能带着这个设置继续。幸运的是，这个通常用于调试、理论上会降低某些 CUDA 操作速度的变量，实际上没有影响吞吐量。我们始终没弄清根因，但如今已经转向不受影响的其他项目，这件事也不再重要。

思路类似前面的停止开关和保存开关，但不再轮询文件，而是监控驻留内存。例如，操作系统显示 CPU 内存只剩 5% 时自动退出：

```python
import psutil
for batch in iterator:
    total_used_percent = psutil.virtual_memory().percent
    if total_used_percent > 0.95:
        print(f"Exiting early since the cpu memory is almost full: ({total_used_percent}%)")
        save_checkpoint()
        sys.exit()

    train_step(batch)
```

GPU 显存也可以采用类似阈值，但需要注意 CUDA 张量缓存和 Python 垃圾回收的时机。要获得实际用量，应先运行垃圾回收，再清空 CUDA 缓存，然后读取统计；显存接近耗尽时，安全退出训练。

```python
import gc
import torch

for batch in iterator:
    gc.collect()
    torch.cuda.empty_cache()

    # get mem usage in GiBs and exit if less than 2GiB of free GPU memory remain
    free, total = map(lambda x: x/2**30, torch.cuda.mem_get_info());
    if free < 2:
        print(f"Exiting early since the GPU memory is almost full: ({free}GB remain)")
        save_checkpoint()
        sys.exit()

    train_step(batch)
```

补充说明：除非确有必要，否则不要这么做，因为缓存能提高速度。理想办法是解决碎片问题。例如，查看 [`PYTORCH_ALLOC_CONF`](https://docs.pytorch.org/docs/stable/notes/cuda.html#environment-variables) 文档中的 `max_split_size_mb`，它控制显存分配方式。[DeepSpeed](https://github.com/deepspeedai/DeepSpeed) 等框架通过启动时预分配张量、反复复用，彻底避免这类碎片问题。

补充说明：这个简化示例适用于单节点。多节点时，需要收集所有参与节点的统计，找出剩余内存最少的节点，并据此采取行动。

## 处理强制作业抢占

前面介绍了用[停止开关](#kill-switch)安全停止训练，适合按需停止或暂停。

HPC 集群上的 SLURM 作业通常有最长运行时间，例如 20 小时。这是因为资源由多个用户或团队共享，每个作业获得一个时间片，到期后被强制停止，让其他作业使用资源。

补充说明：除非作业拥有集群最高优先级，否则很难规划训练完成时间。优先级不最高时，等待数小时甚至数天才能恢复并不少见。

当然，可以让作业被杀掉，寄希望于[上次检查点](#frequent-checkpoint-saving)之后没有运行太久，然后恢复。但这样浪费很大，最好避免。

更高效的方法是在触及硬时间限制、被 SLURM 杀掉之前安全退出。

首先估计安全结束需要多久，通常包含两部分：

1. 如果刚开始一轮迭代，完成它最多需要多久。
2. 保存检查点需要多久。

例如，迭代最多 2 分钟，保存再要 2 分钟，就至少需要 4 分钟宽限。为保险起见，我会至少翻倍。提前一点退出没有坏处，因为不会浪费已有计算。

例如，HPC 允许 100 小时作业，Slurm 脚本中写：

```
#SBATCH --time=100:00:00
```

### 方法 A：启动时告诉程序何时开始退出流程

```bash
srun ... torchrun ... --exit-duration-in-mins 5990
```

100 小时为 6000 分钟，因此这里预留 10 分钟安全退出。

程序启动时创建计时器，每次新迭代前检查是否达到限制；若达到，就保存检查点并退出。

案例：可以查看 [BLOOM 训练作业中的配置](https://github.com/bigscience-workshop/bigscience/blob/58d99c67f643d27b5765a73a2ee2d1ce0a4b2c6b/train/tr11-176B-ml/tr11-176B-ml.slurm#L97-L100)，以及[对应处理代码](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/e52bdabbde3c6895aceb76c1bced295c2646121f/megatron/training.py#L985-L998)：

```python
        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_cuda = torch.cuda.IntTensor(
                [train_time > args.exit_duration_in_mins])
            torch.distributed.all_reduce(
                done_cuda, op=torch.distributed.ReduceOp.MAX)
            done = done_cuda.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             lr_scheduler)
                print_datetime('exiting program after {} minutes'.format(train_time))
                sys.exit()
```

可以看到，由于是分布式训练，必须在所有 rank 之间同步退出事件。

也可以自动计算退出时机，先获取运行中作业的 `EndTime`：

```bash
$ scontrol show -d job $SLURM_JOB_ID | grep Time
   RunTime=00:00:42 TimeLimit=00:11:00 TimeMin=N/A
   SubmitTime=2023-10-26T15:18:01 EligibleTime=2023-10-26T15:18:01
   AccrueTime=2023-10-26T15:18:01
   StartTime=2023-10-26T15:18:01 EndTime=2023-10-26T15:18:43 Deadline=N/A
```

然后在程序里与当前时间比较，设置安全退出窗口。输出中还展示了其他可获取的时间戳和时长。

### 方法 B.1：结束前 X 分钟发送自定义信号

在 sbatch 脚本中设置：

```
#SBATCH --signal=USR1@600
```

SLURM 就会在作业结束前 10 分钟向程序发送 `SIGUSR1`。

补充说明：通常 SLURM 在到期前约 30–60 秒发送 `SIGCONT`+`SIGTERM`；到期时如果仍运行，再发送 `SIGCONT`+`SIGTERM`+`SIGKILL`。虽然可以捕获并处理 `SIGTERM`，但 30 秒不足以让大模型训练安全退出。

演示一下信号发送与捕获。在终端 A 中运行：

```bash
python -c "
import time, os, signal

def sighandler(signum, frame):
    print('Signal handler called with signal', signum)
    exit(0)

signal.signal(signal.SIGUSR1, sighandler)
print(os.getpid())
time.sleep(1000)
"
```

程序会打印进程 PID，例如 `4034989`，然后休眠以模拟工作。在终端 B 中向它发送 `SIGUSR1`：

```bash
kill -s USR1 4034989
```

程序捕获信号，调用 `sighandler`，打印已被调用的信息，然后退出：

```
Signal handler called with signal 10
```

`10` 是 `SIGUSR1` 的数值。

SLURM 中的对应示例如下：

```bash
$ cat sigusr1.slurm
#SBATCH --job-name=sigusr1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:03:00
#SBATCH --partition=mypartition
#SBATCH --output=%x-%j.out
#SBATCH --signal=USR1@170

srun python -c "
import time, os, signal

def sighandler(signum, frame):
    print('Signal handler called with signal', signum)
    exit(0)

signal.signal(signal.SIGUSR1, sighandler)
print(os.getpid())
time.sleep(1000)
"
```

脚本要求 SLURM 在结束前 170 秒发送信号，作业总时间是 180 秒，即 3 分钟。

提交作业：

```bash
sbatch sigusr1.slurm
```

启动 10 秒（`180-170`）后，作业会退出，日志为：

```
58307
Signal handler called with signal 10
```

这说明 PID 为 `58307` 的作业捕获了 `SIGUSR1`（`10`）并退出。

理解机制后，就可以不立即执行 `exit(0)`，而是设置“尽快退出”标志，完成当前迭代，检查标志，保存检查点，再退出。这与[方法 A](#approach-a-tell-the-program-at-launch-time-when-it-should-start-the-exiting-process)中的代码很相似。

### 方法 B.2：选择信号发送给哪个进程

如果 `srun` 启动的不是主训练程序，而是 `torchrun`、`accelerate` 等中间启动器，上述方案可能不工作，因为 `SIGUSR1` 很可能不会由启动器传给子进程。这时需要稍复杂一些的 Slurm 脚本。

将下面这一行：

```
#SBATCH --signal=USR1@600
```

替换为：

```
#SBATCH --signal=B:USR1@600
```

增加的 `B:` 告诉 SLURM，把信号发送给 `sbatch` shell，而不是 `srun` 进程，也就是启动器。

然后将脚本末尾常见的启动器代码：

```bash
CMD="python -u -m torch.distributed.run ... train.py ..." # real command here
LOG_FILE=/path/to/logs/main_log.txt
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_FILE

```

改成：

```bash
trap 'echo "SIGUSR1 received!"; \
pid=$(pgrep -f "^python.*(accelerate|deepspeed|torchrun|distributed.run)"); \
pgrep -P $pid | xargs -r kill -USR1; \
wait;' SIGUSR1

CMD="python -u -m torch.distributed.run ... train.py ..." # real command here
LOG_FILE=/path/to/logs/main_log.txt
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_FILE &

wait
```

由于 `--signal=B:USR1@600` 将信号发给 `sbatch` shell，我们可以捕获并处理它，`trap` 行就是为此设置的。

传给 `trap` 的处理代码找到 `accelerate`、`deepspeed`、`torchrun` 或 `torch.distributed.run` 启动器的直接子进程，再向它们发送 `SIGUSR1`。

最后，为了让 `trap` 工作，需要把 `srun` 放到后台执行，因此在命令末尾加 `&`，同时增加 `wait`，防止 `sbatch` shell 在 `srun` 完成之前退出。

Python 中的信号处理代码与方法 B.1 相同。

把 SLURM 脚本的重要部分放在一起如下：

```bash
$ cat launch.slurm
#!/bin/bash
[...]
#SBATCH --partition=dev
#SBATCH --signal=B:USR1 # Custom preemption signal
[...]

trap 'echo "SIGUSR1 received!"; \
pid=$(pgrep -f "^python.*(accelerate|torchrun|deepspeed|distributed.run)"); \
pgrep -P $pid | xargs -r kill -USR1; wait;' SIGUSR1

CMD="python -u -m torch.distributed.run ... train.py ..." # real command here
LOG_FILE=/path/to/logs/main_log.txt
srun --jobid $SLURM_JOBID bash -c "$CMD" 2>&1 | tee -a $LOG_FILE &

wait
```

原来的训练循环可能是：

```bash
$ cat train.py

for batch in dl:
    train_iteration(batch)
```

现在改为：

```bash
$ cat train.py

import signal
import sys

pre_emption_activated = False
def activate_pre_emption(sig, frame):
    global pre_emption_activated
    print("SIGUSR1 received, saving checkpoint")
    pre_emption_activated = True

signal.signal(signal.SIGUSR1, activate_pre_emption)

for batch in dl:
    train_iteration(batch)

    if pre_emption_activated:
        save_checkpoint()
        sys.exit()
```

实际软件中通常会在 trainer 对象里设置标志，而不是使用 `global`；这里作为简短演示已经足够。

要测试这个方案，把 SLURM 脚本头部改成：

```
#SBATCH --time=0:05:00
#SBATCH --signal=B:USR1@60
```

这里要求作业只运行 5 分钟（`--time=0:05:00`），并在结束前 `60` 秒，也就是启动后 4 分钟，向 `sbatch` 脚本发送 `SIGUSR1`。

### 基于 QoS 的 SLURM 抢占

前面还没有讨论服务质量（QoS）策略，它也可能强制抢占已有作业。机制与“分配的运行时间即将结束”类似，只是可能随时发生，而不是固定在结束前 X 秒。

假设 SLURM 中 `--qos=high` 可以抢占 `--qos=low` 作业，低优先级作业有 10 分钟宽限来关闭：

```bash
$ sacctmgr show qos format=name,priority,preempt,MaxTRESPerUser,GraceTime,Preempt,Flags
      Name   Priority     MaxTRESPU  GraceTime    Preempt                Flags
---------- ---------- ------------- ---------- ---------- --------------------
       low          0                 00:10:00
      high          0                 00:00:00        low
```

这与基于时间的抢占非常相似，只是宽限时间在配置中固定，用户不能修改。

如果以 `--qos=high` 启动作业，而节点不足，SLURM 会终止一些低优先级作业，为高优先级作业腾出资源。

默认 `GraceTime` 可能很短，不足以安全结束程序。这时应请系统管理员延长到符合需求的时间。

除此以外，方法 B.1 和 B.2 的方案也适用于这种强制抢占。
