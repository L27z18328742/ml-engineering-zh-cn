# 检查点

- [torch-checkpoint-convert-to-bf16](./torch-checkpoint-convert-to-bf16)——把已有的 fp32 PyTorch 检查点转换为 bf16。如果发现 [safetensors](https://github.com/safetensors/safetensors) 文件，也会一并转换。这个工具应当很容易改造成其他类似用途。

- [torch-checkpoint-shrink.py](./torch-checkpoint-shrink.py)——有些检查点在保存时，由于某种原因，张量底层 storage 的大小超过了当前视图实际使用的大小。这个脚本通过克隆当前视图并重新保存，让检查点只保留当前视图所需的存储空间。
