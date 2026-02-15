
---

# Evaluation

## Preparation

1. Install [VLMEvalkit](https://github.com/open-compass/VLMEvalKit). For details on installation, please refer to VLMEvalkit_README.md.
```bash
conda create -n eval_video_o3
conda activate eval_video_o3
pip install -e .
```
2. Download benchmark datasets: Video-MME, MLVU, LVBench, LongVideoBench, VideoMMMU, MMVU, and Video-Holmes. Add the dataset paths to the `.env` file.
3. Download the model checkpoints and add their paths to `./vlmeval/config.py`.

## Usage

The complete evaluation scripts can be found in the ./scripts folder.You can start the evaluation by running:

```bash
bash scripts/run_eval.sh
```
