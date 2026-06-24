# Student Model SFT

本文档说明如何将 Seeker-173K 的多轮视频工具轨迹转换为单轮 OPD
轨迹，并完成 Video-o3 学生模型的 SFT 训练、单样本推理和批量推理。

学生模型只接收原始完整视频，并在一次 assistant 输出中生成完整轨迹：

```xml
<think>
<think>reasoning for crop 1</think>
<grounding>{"temporal_segment": [t0, t1], "sampling_strategy": "coarse"}</grounding>
<think>reasoning for crop 2</think>
<grounding>{"temporal_segment": [t2, t3], "sampling_strategy": "fine"}</grounding>
<think>final reasoning</think>
</think>
<answer>A</answer>
```

`<grounding>` 仅声明需要裁剪的原视频时间段。裁剪结果不会在学生模型生成期间
再次输入给学生模型。

## 1. 环境准备

下列命令默认项目根目录为：

```text
OPD
```

```bash
cd Video-o3
```

安装项目内的 LLaMA-Factory：

```bash
cd SFT
conda create -n sft_video_o3 python=3.11
conda activate sft_video_o3
pip install -e ".[torch,metrics]" --no-build-isolation
cd ..
```

准备以下文件：

- Video-o3 基础模型：`../model/Video-o3_SFT_RL`
- Seeker-173K 标注文件
- 标注中引用的原始视频文件

实际路径通过训练和推理 YAML 中的 `model_name_or_path`、`dataset_dir`
和 `media_dir` 配置。

## 2. 构建训练集

### 2.1 可选：筛选本地已有视频的样本

如果 Seeker 标注包含尚未下载的视频，可先筛选视频真实存在的样本：

```bash
python data/build_tiny_trainset.py \
  --input-json ../dataset/Seeker-173K/SFT/sft_llava-video_youtube_qa_mc_2_3_m_clue_multi_w_tool_diff_2790.json \
  --video-dir ../dataset/LLaVA-Video-178K/2_3_m_youtube_v0_1/liwei_youtube_videos/videos/youtube_video_2024 \
  --output-json data/filtered.json
```

### 2.2 转换为单轮 OPD SFT 数据

```bash
python scripts/build_sft_from_seeker.py \
  --input data/filtered.json \
  --output data/student_sft.jsonl
```

如果视频文件完整，也可以直接转换完整 Seeker 标注：

```bash
python scripts/build_sft_from_seeker.py \
  --input ../dataset/Seeker-173K/SFT/sft_llava-video_youtube_qa_mc_2_3_m_clue_multi_w_tool_diff_2790.json \
  --output data/student_sft.jsonl
```

构建少量样本用于调试：

```bash
python scripts/build_sft_from_seeker.py \
  --input data/filtered.json \
  --output data/tiny_student_sft.jsonl \
  --max-samples 4
```

`data/dataset_info.json` 使用名称 `student_sft` 注册
`tiny_student_sft.jsonl`。如果修改输出文件名，也需要同步修改该注册文件。

## 3. 配置训练

训练配置：

```text
SFT/examples/video_o3_tiny_student_sft.yaml
```

运行前重点检查：

```yaml
model_name_or_path: ../../model/Video-o3_SFT_RL
dataset: student_sft
media_dir: ../../dataset/LLaVA-Video-178K/...
output_dir: ../../saves/video-o3-tiny-student-sft/ckpt
```

当前配置用于小规模联调：

```yaml
max_samples: 4
video_maxlen: 8
```

正式训练时应删除 `max_samples` 或将其改为所需样本数，并根据显存和视频长度
调整 `video_maxlen`、`video_max_pixels`、`cutoff_len`、batch size 和梯度累积步数。

## 4. 启动训练

从 `SFT` 目录执行：

```bash
cd SFT
llamafactory-cli train examples/video_o3_tiny_student_sft.yaml
```

指定单张 GPU：

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
  examples/video_o3_tiny_student_sft.yaml
```

多 GPU：

```bash
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  llamafactory-cli train examples/video_o3_tiny_student_sft.yaml
```

默认 checkpoint 输出到：

```text
../../saves/video-o3-tiny-student-sft/ckpt
```

## 5. 单样本推理

推理需要配置另一个环境：

```bash
cd Eval
conda create -n eval_video_o3 python=3.11
conda activate eval_video_o3
pip install -e .
cd ..
```

`scripts/test_sft.py` 读取 JSONL 的第一条样本，只调用一次 `generate()`，
并使用正式 OPD parser 检查输出格式。

从项目根目录执行：

```bash
python scripts/test_sft.py \
  --model-path ../saves/video-o3-tiny-student-sft/ckpt \
  --jsonl data/tiny_student_sft.jsonl \
  --media-dir ../dataset/LLaVA-Video-178K/2_3_m_youtube_v0_1/liwei_youtube_videos/videos/youtube_video_2024 \
  --dtype bf16 \
  --max-new-tokens 1536 \
  --video-nframes 128
```

输出末尾会显示：

```text
========== FORMAT CHECK ==========
PASSED
```

## 6. 批量推理

批量推理配置：

```text
SFT/examples/video_o3_tiny_student_sft_infer.yaml
```

从 `SFT` 目录执行：

```bash
cd SFT
llamafactory-cli train examples/video_o3_tiny_student_sft_infer.yaml
```

这里仍使用 `train` 子命令，因为 LLaMA-Factory 的 SFT workflow 根据：

```yaml
do_train: false
do_predict: true
predict_with_generate: true
```

进入批量生成流程。

默认推理参数为：

```yaml
per_device_eval_batch_size: 1
do_sample: false
num_beams: 1
max_new_tokens: 1536
```

视频推理显存占用较大，建议先保持 batch size 为 1。批量推理结果保存在：

```text
../../saves/video-o3-tiny-student-sft/infer/generated_predictions.jsonl
```

每一行包含：

```json
{"prompt": "...", "predict": "...", "label": "..."}
```

- `prompt`：模型输入
- `predict`：学生模型生成的单轮 OPD 轨迹
- `label`：SFT 数据中的参考轨迹

## 7. 常见问题

### 找不到数据集

确认 YAML 中使用的是注册名：

```yaml
dataset: student_sft
```

训练使用 `dataset`，批量推理使用：

```yaml
eval_dataset: student_sft
```

### 找不到视频

确认 `media_dir` 指向视频目录，并且 JSONL 中 `videos[].url` 的文件名存在于
该目录。

### CUDA 显存不足

依次尝试：

1. 保持 `per_device_train_batch_size` 或 `per_device_eval_batch_size` 为 1。
2. 减小 `video_maxlen` 和 `video_max_pixels`。
3. 减小 `cutoff_len`。
4. 增加 `gradient_accumulation_steps`。
5. 使用 DeepSpeed 配置。

### 生成格式检查失败

检查生成是否被 `max_new_tokens` 截断，并确认输出满足：

- 只有一个外层 `<think>...</think>`
- 每一步都有内层 `<think>...</think>`
- grounding 是严格 JSON
- `<answer>...</answer>` 位于外层 `</think>` 之后
- `</answer>` 后没有额外内容
