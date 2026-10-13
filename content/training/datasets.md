# 处理数据集

## 在主进程中预处理和缓存数据集

HF Accelerate 提供了非常方便的上下文管理器 [`main_process_first`](https://huggingface.co/docs/accelerate/v0.4.0/accelerator.html#accelerate.Accelerator.main_process_first)，可以这样编写代码：

```python
with accelerator.main_process_first():
    # load and pre-process datasets
    dataset = datasets.load_dataset(...)
    # optionally cache it and have the rest of the processes load the cache
```

从而代替下面这种不够直观、需要重复编写代码的方式：

```python
if rank == 0:
    dataset = datasets.load_dataset(...)
dist.barrier()
if not rank == 0:
    dataset = datasets.load_dataset(...)
```

应该在主进程中下载和处理数据，而不是让所有进程都执行这些操作。否则，各进程会并行重复相同的工作，而且很可能写入同一个位置，导致输出相互交错并损坏。从 I/O 的角度来看，串行执行这类工作通常也快得多。

这里有 `main_process_first` 和 `local_main_process_first` 两种上下文管理器。前者适用于数据位于共享文件系统、所有计算节点都能访问的情况；后者适用于数据保存在每个节点本地的情况。

如果没有使用 HF Accelerate，我也实现了类似的上下文管理器，只是名称不同：

- `global_main_process_first`：用于共享文件系统。
- `local_main_process_first`：用于节点本地文件系统。

可以在[这里](tools/main_process_first.py)找到它们。

如果希望编写能自动适配共享和本地文件系统的通用代码，该怎么办？我还增加了一个辅助函数，自动识别所使用的文件系统类型，并据此调用合适的上下文管理器。它叫作 `main_process_by_path_first`，用法如下：

```python
path = "/path/to/data"
with main_process_by_path_first(path):
    # load and pre-process datasets
    dataset = datasets.load_dataset(...)
    # optionally cache it and have the rest of the processes load the cache
```

可以在[这里](tools/main_process_first.py)找到它。

除了上下文管理器，当然还需要判断主进程类型的工具函数。因此，也提供了三个与上述管理器对应的函数：

- `is_main_process_by_path(path)`
- `is_local_main_process()`
- `is_global_main_process()`

这些函数都在[这里](tools/main_process_first.py)。

运行下面的命令即可查看它们的实际效果：

```bash
python -u -m torch.distributed.run --nproc_per_node=2 --rdzv_endpoint localhost:6000  --rdzv_backend c10d tools/main_process_first.py
```
