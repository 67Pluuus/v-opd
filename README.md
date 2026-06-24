# O3-OPD

Video-o3 的 On-Policy Distillation 训练实现。OPD 代码复用仓库现有的
Video-o3/Qwen2.5-VL 模型、chat template 和视频处理逻辑。

学生模型的 SFT 冷启动说明见
[README_STUDENT_SFT.md](README_STUDENT_SFT.md)。

## 训练逻辑

每个 OPD step 执行：

1. student 根据原始完整视频和问题，用当前参数一次生成完整轨迹；
2. 严格解析 `<think>`、`<grounding>` 和 `<answer>`，非法轨迹只保留合法前缀；
3. 根据 student 自己选择的时间段动态构造 teacher 多轮观察上下文；
4. student 在单轮累计前缀下、teacher 在原视频及裁剪视频上下文下对相同 target token 做 teacher forcing；
5. 在完整词表上计算精确的 `KL(teacher || student)`，只更新 student。

## 1. 构建 SFT 冷启动数据

```bash
python scripts/build_sft_from_seeker.py \
  --input ../dataset/Seeker-173K/SFT/sft_llava-video_youtube_qa_mc_2_3_m_clue_multi_w_tool_diff_2790.json \
  --output data/student_sft.jsonl
```

然后按 `README_STUDENT_SFT.md` 训练 student。

## 2. 构建 OPD 数据

如果 Seeker 标注包含尚未下载的视频，可先筛选视频真实存在的样本，参考`README_STUDENT_SFT.md`，然后运行：

```bash
python scripts/build_opd_from_seeker.py \
  --input data/filtered.json \
  --output data/tiny_opd_train.jsonl
```

视频完整则运行：

```bash
python scripts/build_opd_from_seeker.py \
  --input ../dataset/Seeker-173K/SFT/sft_llava-video_youtube_qa_mc_2_3_m_clue_multi_w_tool_diff_2790.json \
  --output data/opd_train.jsonl
```

OPD 样本分别保存：

- `student_messages`：要求单次输出完整轨迹的 student prompt；
- `teacher_messages`：原始 Video-o3 多轮工具 prompt；
- `videos`：原始完整视频；
- `student_target`：仅用于数据检查，不参与正常 on-policy 训练。


## 3. 配置

生产训练入口直接读取 [configs/opd_small.yaml](configs/opd_small.yaml)，包括：

- student/teacher checkpoint；
- 数据集和视频目录；
- 原视频帧数及裁剪视频 FPS；
- coarse/medium/fine 的动态视觉 token 配额；
- generation 参数；
- batch size、梯度累积、优化器、scheduler；
- checkpoint 保存、保留数量和断点恢复；
- mixed precision 和 DeepSpeed 配置。

裁剪视频使用 `crop_fps`，不会再被原视频的固定 `nframes` 覆盖。

## 4. 单卡训练

安装 SFT/Video-o3 依赖后：

```bash
python scripts/train_opd.py \
  --config configs/opd_small.yaml \
  --reverse-kl-exact
```

最终可直接推理的模型和 processor 保存到 `train.output_dir`。训练中间状态保存为：

```text
output_dir/
  checkpoint-20/
  checkpoint-40/
  opd_config.json
  opd_trainer_state.json
```

中间 checkpoint 包含 Accelerate/DeepSpeed student 状态、optimizer、
scheduler、随机数状态、processor 和训练进度；其内部文件布局取决于是否启用
DeepSpeed。中间目录主要用于恢复训练，最终 `output_dir` 用于推理。

## 5. DDP

Accelerate 会自动将同一训练入口包装成 DDP：

```bash
accelerate launch --multi_gpu --num_processes 8 \
  scripts/train_opd.py \
  --config configs/opd_small.yaml \
  --reverse-kl-exact
```


## 6. DeepSpeed

在 YAML 中设置：

```yaml
train:
  deepspeed: SFT/examples/deepspeed/ds_z2_config.json
```

然后仍使用 Accelerate 启动：

```bash
accelerate launch --multi_gpu --num_processes 8 \
  scripts/train_opd.py \
  --config configs/opd_small.yaml \
  --reverse-kl-exact
```

仓库现有的 ZeRO-2、ZeRO-3 和 offload 配置位于
`SFT/examples/deepspeed/`。teacher 始终冻结，每个进程保留一份 teacher；
DeepSpeed 负责 student、optimizer 和梯度状态。

## 7. 断点恢复

恢复指定 checkpoint：

```bash
accelerate launch scripts/train_opd.py \
  --config configs/opd_small.yaml \
  --reverse-kl-exact \
  --resume-from-checkpoint ../saves/opd-small/checkpoint-40
```

恢复最新 checkpoint：

```bash
accelerate launch scripts/train_opd.py \
  --config configs/opd_small.yaml \
  --reverse-kl-exact \
  --resume-from-checkpoint latest
```

数据 sampler 由 `seed + epoch` 确定，恢复时会跳过已经完成的 microbatch。

## 8. 测试 OPD 模型

`test_opd.py` 加载最终 OPD student，生成一条 on-policy 轨迹，并检查：

- 视频时长；
- grounding 是否越界；
- 标签和 JSON 格式；
- 可拆分出的 teacher task 数；
- 最终 answer。

```bash
python scripts/test_opd.py \
  --dataset data/opd_train.jsonl \
  --model-path ../saves/opd-small \
  --media-dir ../dataset/LLaVA-Video-178K/2_3_m_youtube_v0_1/liwei_youtube_videos/videos/youtube_video_2024 \
  --sample-index 0
```

确定性推理是默认行为；如需采样可增加 `--do-sample`。

## 实现说明

- 视频时长按 `decord → PyAV → OpenCV → ffprobe` 顺序自动探测并缓存；
- student/teacher 的 target token id 和词表大小必须一致，否则精确 KL 直接报错；
- 精确 KL 按 task 流式累加，不拼接所有 `[sequence, vocabulary]` logits；
- checkpoint 使用 Accelerate 原生状态保存，因此支持单卡、DDP 和 DeepSpeed 恢复
