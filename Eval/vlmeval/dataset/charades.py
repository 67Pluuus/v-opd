from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension, LMUDataRoot
from .video_base import VideoBaseDataset
from .utils.cgbench import post_process, get_timestampes


class Charades(VideoBaseDataset):
    """
    Charades 时序定位数据集。
    仅包含基于 IoU 的时间区间预测任务，模型需要输出动作发生的时间片段。
    """

    TYPE = "Video-Temporal-Grounding"
    SYS = (
        "You will be provided with uniformly sampled frames from a video, along with "
        "a description of the action that happens in the video.\n"
        "Identify the time intervals (in seconds) where the described action occurs.\n"
        "Only output the answer in the following format:\n\n"
        '```json\n{"result": [[start1, end1], [start2, end2], ...]}\n```\n\n'
        "Each interval is in seconds. Provide at least one interval and at most five."
    )

    def __init__(self, dataset="Charades", use_frame_time=True, nframe=-1, fps=2, frames_limit=768):
        """
        Args:
            dataset (str): 数据集名称。
            use_frame_time (bool): 是否在提示中附带采样帧对应的时间戳。
            nframe (int): 采样帧数（与 fps 互斥）。
            fps (int): 采样帧率（与 nframe 互斥）。
        """
        self.use_frame_time = use_frame_time
        self.dataset_name = dataset
        self.frames_limit = frames_limit
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ["Charades"]

    def _frame_paths(self, key, num_frames):
        frame_root = osp.join(self.frame_root, key)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]

    def _frame_paths_fps(self, key, num_frames, fps):
        frame_root = osp.join(self.frame_root, key)
        os.makedirs(frame_root, exist_ok=True)
        return [
            osp.join(frame_root, self.frame_tmpl_fps.format(i, num_frames, fps))
            for i in range(1, num_frames + 1)
        ]

    def prepare_dataset(
        self,
        dataset="Charades",
        json_path=None,
        root_dir=None,
    ):
        if json_path is None:
            json_path = os.getenv('Charades_json_PATH', None)
        if json_path is None:
            raise ValueError('Charades_json_PATH is not set')

        if root_dir is None:
            root_dir = os.getenv('Charades_PATH', None)
        if root_dir is None:
            raise ValueError('Charades_PATH is not set')

        assert osp.exists(json_path), f"{json_path} 不存在，请检查数据路径。"

        tsv_path = "Charades.tsv"

        data = load(json_path)

        def extract_intervals(item):
            solution = item.get("solution", {})
            clue = solution.get("clue", []) if isinstance(solution, dict) else []
            intervals = []
            for c in clue:
                ts = c.get("timestamp", None)
                if isinstance(ts, list) and len(ts) == 2:
                    intervals.append([float(ts[0]), float(ts[1])])
            return intervals

        df = pd.DataFrame(data)
        df = df.assign(index=range(len(df)))
        df["question"] = df["problem"]
        df["clue_intervals"] = df.apply(extract_intervals, axis=1)
        df["answer"] = df["clue_intervals"]
        df["task_mode"] = "miou"

        df = df[
            [
                "index",
                "doc_id",
                "video",
                "duration",
                "question",
                "answer",
                "clue_intervals",
                "task_mode",
                "data_source",
            ]
        ]

        df.to_csv(tsv_path, sep="\t", index=False)

        # 默认使用工作区路径作为视频根目录，可通过环境变量覆盖
        dataset_path = root_dir

        return dict(data_file=tsv_path, root=dataset_path)

    def save_video_frames(self, video_path, key):
        import decord

        vid_path = osp.join(self.data_root,"data", video_path)
        vid = decord.VideoReader(vid_path)
        video_fps = vid.get_avg_fps()
        n_frames = len(vid)
        # 视频原始信息
        video_info = {
            "fps": video_fps,
            "n_frames": n_frames,
            "duration": n_frames / video_fps,
        }

        if self.nframe > 0 and self.fps < 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self._frame_paths(key, len(indices))
            # 保存采样信息
            video_info["sample_fps"] = video_fps / step_size if step_size > 0 else video_fps
            video_info["sample_n_frame"] = len(indices)
        else:
            # 使用 fps 采样
            required_frames = max(1, int(video_info["duration"] * self.fps))
            if required_frames > self.frames_limit:
                step_size = n_frames / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                # frame_paths = self._frame_paths_fps(key, len(indices), self.fps)
                frame_root = osp.join(self.frame_root, video)
                os.makedirs(frame_root, exist_ok=True)
                frame_paths = [osp.join(frame_root, self.frame_tmpl.format(i, self.frames_limit)) for i in range(1, self.frames_limit + 1)]
                # 保存采样信息
                video_info["sample_fps"] = self.frames_limit / video_info["duration"]
                video_info["sample_n_frame"] = self.frames_limit
            else:
                step_size = video_fps / self.fps
                indices = [int(i * step_size) for i in range(required_frames)]
                frame_paths = self._frame_paths_fps(key, len(indices), self.fps)
                # 保存采样信息
                video_info["sample_fps"] = self.fps
                video_info["sample_n_frame"] = required_frames

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            if not np.all([osp.exists(p) for p in frame_paths]):
                images = [Image.fromarray(vid[i].asnumpy()) for i in indices]
                for im, pth in zip(images, frame_paths):
                    if not osp.exists(pth):
                        # print("pth: ", pth)
                        im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, video_llm=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        video_path = line["video"]
        video_key = str(line.get("doc_id", line.get("index", "charades")))

        # message = [dict(type="system", value=self.SYS)]
        message = []
        frames, indices, video_info = self.save_video_frames(video_path, key=video_key)

        if video_llm:
            actual_fps = (
                self.frames_limit / video_info["duration"]
                if len(frames) == self.frames_limit and video_info["duration"] > 0
                else self.fps
            )
            message.append(
                dict(
                    type="video",
                    value=frames,
                    sample_fps=actual_fps,
                    # min_pixels=1 * 2 * 2 * 16 * 16,
                    # max_pixels=640 * 32 * 32,
                    # total_pixels=224000 * 4 * 16 * 16,
                )
            )
        else:
            message.extend(dict(type="image", value=im) for im in frames)

        user_prompt = ""
        
        # 加上time instruction
        user_prompt += f"This video is uniformly sampled at {video_info['sample_fps']:.2f} fps, contains {video_info['sample_n_frame']:.1f} frames from 0 seconds to {video_info['duration']:.1f} seconds.\n"

        user_prompt += f"Please analyze the provided video and locate the video segment according to the given query.\nQuery: {line['question']}"

        message.append(dict(type="text", value=user_prompt))

        return message

    def evaluate(self, eval_file, **judge_kwargs):
        assert get_file_extension(eval_file) in ["xlsx", "json", "tsv"], "仅支持 xlsx/json/tsv 评测文件。"

        tgt_file = get_intermediate_file_path(eval_file, "_rating", "json")
        score_file = get_intermediate_file_path(eval_file, "_score")

        data = load(eval_file)

        if "task_mode" not in data:
            data["task_mode"] = "miou"

        def normalize_prediction(pred_raw):
            """兼容 {"temporal_segment": [s, e]} / {"result": [...]} / 直接数字对等多种输入。"""
            if pred_raw is None or (isinstance(pred_raw, float) and np.isnan(pred_raw)):
                return pred_raw

            text = str(pred_raw)

            # 去掉 Markdown 代码块包裹
            if "```" in text:
                first = text.find("```")
                last = text.rfind("```")
                if first != -1 and last != -1 and last > first:
                    text = text[first + 3:last].strip()
                if text.startswith("json"):
                    text = text[4:].strip()

            # JSON 尝试解析
            try:
                obj = json.loads(text)
            except Exception:
                return text  # 留给后续 regex 兜底

            # 已经是 {"result": [[s,e], ...]}
            if isinstance(obj, dict) and "result" in obj:
                return json.dumps(obj)

            # {"temporal_segment": [s, e]} 或 [["s","e"]]
            if isinstance(obj, dict) and "temporal_segment" in obj:
                seg = obj["temporal_segment"]
                if isinstance(seg, list) and len(seg) == 2:
                    try:
                        s, e = float(seg[0]), float(seg[1])
                        return json.dumps({"result": [[s, e]]})
                    except Exception:
                        return text

            # 如果是单个区间数组 [s, e]
            if isinstance(obj, list) and len(obj) == 2 and all(isinstance(x, (int, float, str)) for x in obj):
                try:
                    s, e = float(obj[0]), float(obj[1])
                    return json.dumps({"result": [[s, e]]})
                except Exception:
                    return text

            return text

        data_un = data[~pd.isna(data["prediction"])].copy()
        data_pred_na = data[pd.isna(data["prediction"])].copy()

        data_pred_na["score"] = -1

        data_un["score"] = data_un.apply(
            lambda row: post_process(
                response=normalize_prediction(row["prediction"]),
                right_answer=row["answer"],
                task_mode="miou",
                duration=row.get("duration", 0),
            ),
            axis=1,
        )

        data = pd.concat([data_pred_na, data_un])
        rejected = (data["score"] == -1).sum()

        print(
            f"Among {len(data)} questions, failed to obtain prediction for {len(data_pred_na)} questions, "
            f"failed to obtain the score for {rejected - len(data_pred_na)} questions. "
            f"Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating."
        )

        dump(data, score_file)

        valid = data[data["score"] != -1]
        overall = round(valid["score"].mean(), 4) if len(valid) else 0
        
        # 计算不同 IoU 阈值下的 Recall@1（基于所有样本，包括失败的）
        r1_03 = (data["score"] >= 0.3).sum() if len(data) > 0 else 0
        r1_05 = (data["score"] >= 0.5).sum() if len(data) > 0 else 0
        r1_07 = (data["score"] >= 0.7).sum() if len(data) > 0 else 0
        
        # 计算百分比（使用总数作为分母）
        total = len(data)
        r1_03_pct = round(r1_03 / total * 100, 2) if total > 0 else 0
        r1_05_pct = round(r1_05 / total * 100, 2) if total > 0 else 0
        r1_07_pct = round(r1_07 / total * 100, 2) if total > 0 else 0
        
        rating = {
            "overall": overall,
            "count": len(valid),
            "total": len(data),
            "R@1_IoU=0.3": r1_03_pct,
            "R@1_IoU=0.5": r1_05_pct,
            "R@1_IoU=0.7": r1_07_pct,
        }
        
        print(f"\n=== Charades-STA Evaluation Results ===")
        print(f"Mean IoU: {overall} (on valid samples only)")
        print(f"R@1, IoU=0.3: {r1_03}/{total} ({r1_03_pct}%)")
        print(f"R@1, IoU=0.5: {r1_05}/{total} ({r1_05_pct}%)")
        print(f"R@1, IoU=0.7: {r1_07}/{total} ({r1_07_pct}%)")
        print(f"Valid samples: {len(valid)}/{total}")
        print(f"=======================================\n")

        dump(rating, tgt_file)
        return rating

