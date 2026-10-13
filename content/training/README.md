# 训练

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
