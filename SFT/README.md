
---

# SFT

## Preparation

1. Install [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)
```bash
conda create -n sft_video_o3 python=3.11
conda activate sft_video_o3
pip install -e ".[torch,metrics]" --no-build-isolation
# Optional: flash-attn
# Download the appropriate .whl from https://github.com/Dao-AILab/flash-attention, then:
# pip install flash-attn.whl
# Optional: liger
# pip install liger-kernel
```
2. Download the SFT dataset (To be released)
3. Download the [Qwen2.5VL](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) model weights

## Training

1. Before training the model, we recommend preprocessing the dataset first.
2. The configuration files for the two-stage SFT training can be found in the `./examples` directory. Please refer to LLaMA-Factory for training, for example:

### Stage-1:

```bash
llamafactory-cli train examples/video_o3_sft_stage1.yaml
```

### Stage-2:

```bash
llamafactory-cli train examples/video_o3_sft_stage2.yaml
```