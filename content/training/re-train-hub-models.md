# 利用微调示例从头训练 HF Hub 上的模型

HF Transformers 提供了出色的微调示例：https://github.com/huggingface/transformers/tree/main/examples/pytorch 。这些示例几乎涵盖所有模态，而且开箱即用。

**但如果想从头训练，而不是微调，该怎么办？**

下面介绍一个简单的变通方法。

这里以 `facebook/opt-1.3b` 为例，并计划采用 bf16 训练：

```bash
cat << EOT > prep-bf16.py
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

mname = "facebook/opt-1.3b"

config = AutoConfig.from_pretrained(mname)
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(mname)

path = "opt-1.3b-bf16"

model.save_pretrained(path)
tokenizer.save_pretrained(path)
EOT
```

然后运行：

```bash
python prep-bf16.py
```

这会创建 `opt-1.3b-bf16` 文件夹，包含从头训练模型所需的一切。换句话说，你得到了一个形式上类似预训练模型的模型，但它只完成了初始化，还没有进行任何训练。

如果计划使用 `torch.float16` 或 `torch.float32`，相应修改上面的脚本即可。

现在可以像平常微调一样，继续训练这个已经保存的模型：

```bash
torchrun \
--nproc_per_node=1 --nnode=1 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=9901 \
examples/pytorch/language-modeling/run_clm.py --bf16 \
--seed 42 --model_name_or_path opt-1.3b-bf16 \
--dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
--per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
--gradient_accumulation_steps 1 --do_train --do_eval --logging_steps 10 \
--save_steps 1000 --eval_steps 100 --weight_decay 0.1 --num_train_epochs 1 \
--adam_beta1 0.9 --adam_beta2 0.95 --learning_rate 0.0002 --lr_scheduler_type \
linear --warmup_steps 500 --report_to tensorboard --output_dir save_dir
```

其中关键参数是：

```
--model_name_or_path opt-1.3b-bf16
```

这里的 `opt-1.3b-bf16` 就是上一步生成的本地目录。

有时可以找到原模型训练时使用的相同数据集，有时则必须使用替代数据集。

其余超参数通常可以在模型随附的论文或文档中找到。

总之，这种方法让你能够利用微调示例，从头训练 [HF Hub](https://huggingface.co/models) 上找到的模型。
