from __future__ import annotations


SYSTEM_PROMPT_MULTI_ROUND_MC="""You are a helpful assistant. Answer the user's multiple-choice question based on the provided video.
Output your thinking process within the `<think>` and `</think>` tags.
If you find any video segments that might help answer your questions, you can view a specific area in detail by outputting `<grounding>{\"temporal_segment\": [t0, t1], \"sampling_strategy\": \"medium\"}</grounding>`, where t0 and t1 are the start and end times (in integer seconds) of the video segment you want to observe in detail within the entire video, sampling_strategy must be a string and in ['coarse', 'fine', 'medium'].
Once you believe you have observed all the video clues that help you answer the question and are able to integrate all the existing clues to give the correct answer, you should put the correct option letter within the `<answer>` and `</answer>` tags.
The following is a example of the thinking process and the final answer:
 - When you want to examine a specific clue segment of a video more closely, produce exactly: '<think>your thinking process here</think>\n<grounding>{"temporal_segment": [START_TIMESTAMP, END_TIMESTAMP], "sampling_strategy": "medium"}</grounding>\n'.
 - When you believe the current information is sufficient to give an answer, produce exactly: '<think>your thinking process here</think>\n<answer>Option Letter</answer>\n'.
"""

VIDEO_INSERT_PROMPT = """Here is the original full video (Observation 0):
<video>This video is uniformly sampled at {sample_fps:.2f} fps, contains {total_frames:.1f} frames from 0 seconds to {max_duration:.1f} seconds."""

QUESTION_TEMPLATE_MULTI_ROUND_MC = """
Answer the following multiple-choice question according to the content of the video: 
{question}
Options:
{options}
You are advised to first observe potential clue segments, use '<think>YOUR THINK</think>\n<grounding>YOUR GROUNDING</grounding>\n' to to specify the segment you want to observe in detail.
If the evidence is visible in the original video for a long enough time and is clear enough to support a confident and correct answer, you may answer directly.
"""

QUESTION_TEMPLATE_MULTI_ROUND_MC_LVBench = """
Answer the following multiple-choice question according to the content of the video: 
{question}
You are advised to first observe potential clue segments, use '<think>YOUR THINK</think>\n<grounding>YOUR GROUNDING</grounding>\n' to to specify the segment you want to observe in detail.
If the evidence is visible in the original video for a long enough time and is clear enough to support a confident and correct answer, you may answer directly.
"""

QUESTION_TEMPLATE_MULTI_ROUND_MC_LVBench_With_Subtitle = """
{subtitle}
Answer the following multiple-choice question according to the content of the video: 
{question}
You are advised to first observe potential clue segments, use '<think>YOUR THINK</think>\n<grounding>YOUR GROUNDING</grounding>\n' to to specify the segment you want to observe in detail.
If the evidence is visible in the original video for a long enough time and is clear enough to support a confident and correct answer, you may answer directly.
"""

# QUESTION_TEMPLATE_MULTI_ROUND_TG_CGBench_Charades = """
# {question}
# You are advised to first observe potential clue segments, use '<think>YOUR THINK</think>\n<grounding>YOUR GROUNDING</grounding>\n' to to specify the segment you want to observe in detail.
# If the evidence is visible in the original video for a long enough time and is clear enough to support a confident and correct answer, you may answer directly.
# """

QUESTION_TEMPLATE_MULTI_ROUND_OE_VideoMMMU = """
{pre_prompt}{question}
You are advised to first observe potential clue segments, use '<think>YOUR THINK</think>\n<grounding>YOUR GROUNDING</grounding>\n' to to specify the segment you want to observe in detail.
If the evidence is visible in the original video for a long enough time and is clear enough to support a confident and correct answer, you may answer directly.
"""

QUESTION_TEMPLATE_MULTI_ROUND_MC_VideoMMMU = """
{pre_prompt}{question}{candidates}{post_prompt}
You are advised to first observe potential clue segments, use '<think>YOUR THINK</think>\n<grounding>YOUR GROUNDING</grounding>\n' to to specify the segment you want to observe in detail.
If the evidence is visible in the original video for a long enough time and is clear enough to support a confident and correct answer, you may answer directly.
"""

QUESTION_TEMPLATE_MULTI_ROUND_MC_VideoMMMU_adaptation = """
{pre_prompt}{question}{candidates}{post_prompt}
You are advised to first observe potential clue segments, use '<think>YOUR THINK</think>\n<grounding>YOUR GROUNDING</grounding>\n' to to specify the segment you want to observe in detail.
If the evidence is visible in the original video for a long enough time and is clear enough to support a confident and correct answer, you may answer directly.
"""

QUESTION_TEMPLATE_MULTI_ROUND_OE_MMVU = """
Answer the following question according to the content of the video: {question}
You are advised to first observe potential clue segments, use '<think>YOUR THINK</think>\n<grounding>YOUR GROUNDING</grounding>\n' to to specify the segment you want to observe in detail.
If the evidence is visible in the original video for a long enough time and is clear enough to support a confident and correct answer, you may answer directly.
"""



TOOL_CALL_CROP_VIDEO_MULTI_TRUN_PROMPT_MC = (
    "After the above Action {action_turn}, here is the refined video clip (Observation {observation_turn}):\n"
    "<video>\n"
    "Continue your reasoning process inside <think> and </think>. If needed, you can keep selecting temporal "
    "segments from the original video by outputting <grounding> and </grounding> as before. Once you are ready "
    "to provide the final answer, put the selected option letter inside <answer> and </answer>."
)



class VideoO3PromptMixin:
    """
    Mixin class for Qwen2VLChat to build custom prompt for different datasets.

    Requires the following methods to be implemented in the subclass:
        - dump_image(line, dataset: str) -> str | list[str]

    Implements the following methods:
        - use_custom_prompt(dataset: str) -> bool
        - build_prompt(line, dataset: str) -> list[dict[str, str]]
    """

    def __init__(self, *args, use_custom_prompt: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._use_custom_prompt = use_custom_prompt

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def use_custom_prompt(self, dataset: str) -> bool:
        from vlmeval.dataset import DATASET_TYPE
        dataset_type = DATASET_TYPE(dataset, default=None)

        if not self._use_custom_prompt:
            return False
        else:
            return True
        

    def build_prompt(self, line, dataset: str, video_llm: bool = False) -> list[dict[str, str]]:

        return self.build_o3_prompt(line, dataset)
        # from vlmeval.dataset import DATASET_TYPE

        # if dataset in {'MMMU_DEV_VAL', 'MMMU_TEST'}:
        #     return self._build_mmmu_prompt(line, dataset)
        # dataset_type = DATASET_TYPE(dataset, default=None)
        # if dataset_type == 'MCQ':
        #     return self._build_mcq_prompt(line, dataset)
        # if dataset_type == 'Y/N':
        #     return self._build_yorn_prompt(line, dataset)
        # if dataset_type == 'VQA':
        #     return self._build_vqa_prompt(line, dataset)
        # raise ValueError(f'Unsupported dataset: {dataset}')

    def build_o3_prompt(self, line, dataset: str) -> list[dict[str, str]]:


        use_frames = True

        info = dataset.dump_prompt_info(line,use_frames)

        message = []
        # 作为raw_video的路径，用于后续步骤的crop 
        message.append(dict(type='raw_video', value=info['video']))
        
        control_insert_pos = len(message)

        prompt = VIDEO_INSERT_PROMPT


        # 可以读视频也可以读帧，更快
        if use_frames:
            message.append(dict(
                type='video', 
                value=info['frames'],
                min_pixels=info.get('min_pixels', None),
                max_pixels=info.get('max_pixels', None),
                total_pixels=info.get('total_pixels', None),
                ))
            if len(info['frames']) == 0:
                print("Warning: No frames found for video: ", info['video'])
        else:
            message.append(dict(
                type='video', 
                value=info,
                min_pixels=info.get('min_pixels', None),
                max_pixels=info.get('max_pixels', None),
                total_pixels=info.get('total_pixels', None),
                ))
        
        # 最后一帧单独添加一下
        if 'VideoMMMU' in dataset.dataset_name and info['category'] == 'Adaptation':
            prompt = prompt.replace("<video>","<video><image>")
            message.append(dict(type='image', value=info['image']))

        # 格式化 prompt 模板
        need_oe = False
        need_tg = False
        if 'Video-MME' in dataset.dataset_name or 'Video_Holmes' in dataset.dataset_name:
            prompt += QUESTION_TEMPLATE_MULTI_ROUND_MC
            prompt = prompt.format(sample_fps=info['sample_fps'], total_frames=info['sample_n_frames'], max_duration=info['duration'], question=info['question'], options=info['candidates'])
        elif 'LVBench' in dataset.dataset_name or 'LongVideoBench' in dataset.dataset_name:
            if info.get('subtitle', None) is not None:
                prompt += QUESTION_TEMPLATE_MULTI_ROUND_MC_LVBench_With_Subtitle
                prompt = prompt.format(sample_fps=info['sample_fps'], total_frames=info['sample_n_frames'], max_duration=info['duration'], subtitle=info['subtitle'], question=info['question'])
            else:
                prompt += QUESTION_TEMPLATE_MULTI_ROUND_MC_LVBench
                prompt = prompt.format(sample_fps=info['sample_fps'], total_frames=info['sample_n_frames'], max_duration=info['duration'], question=info['question'])
        elif 'VideoMMMU' in dataset.dataset_name:
            # prompt = QUESTION_TEMPLATE_MULTI_ROUND_MC_VideoMMMU.format(question=info['question'])
            if info['question_type'] == 'multiple-choice':
                if info['category'] == 'Adaptation':
                    prompt += QUESTION_TEMPLATE_MULTI_ROUND_MC_VideoMMMU_adaptation
                    prompt = prompt.format(sample_fps=info['sample_fps'], total_frames=info['sample_n_frames'], max_duration=info['duration'], pre_prompt=info['pre_prompt'], question=info['question'], candidates=info['candidates'], post_prompt=info['post_prompt'])
                else:
                    prompt += QUESTION_TEMPLATE_MULTI_ROUND_MC_VideoMMMU
                    prompt = prompt.format(sample_fps=info['sample_fps'], total_frames=info['sample_n_frames'], max_duration=info['duration'], pre_prompt=info['pre_prompt'], question=info['question'], candidates=info['candidates'], post_prompt=info['post_prompt'])
            else:
                prompt += QUESTION_TEMPLATE_MULTI_ROUND_OE_VideoMMMU
                prompt = prompt.format(sample_fps=info['sample_fps'], total_frames=info['sample_n_frames'], max_duration=info['duration'], pre_prompt=info['pre_prompt'], question=info['question'])
                need_oe = True
        elif 'MMVU' in dataset.dataset_name:
            if info['question_type'] == 'multiple-choice':
                prompt += QUESTION_TEMPLATE_MULTI_ROUND_MC
                prompt = prompt.format(sample_fps=info['sample_fps'], total_frames=info['sample_n_frames'], max_duration=info['duration'], question=info['question'], options=info['candidates'])
            else:
                prompt += QUESTION_TEMPLATE_MULTI_ROUND_OE_MMVU
                prompt = prompt.format(sample_fps=info['sample_fps'], total_frames=info['sample_n_frames'], max_duration=info['duration'], question=info['question'])
                need_oe = True

        elif 'MLVU' in dataset.dataset_name:
            if info['question_type'] == 'multiple-choice':
                prompt += QUESTION_TEMPLATE_MULTI_ROUND_MC_LVBench
                prompt = prompt.format(sample_fps=info['sample_fps'], total_frames=info['sample_n_frames'], max_duration=info['duration'], question=info['question'])
            else:
                prompt += QUESTION_TEMPLATE_MULTI_ROUND_OE_MMVU
                prompt = prompt.format(sample_fps=info['sample_fps'], total_frames=info['sample_n_frames'], max_duration=info['duration'], question=info['question'])
                need_oe = True


        else:
            raise ValueError(f'Unsupported dataset: {dataset.dataset_name}')
        
        if need_oe:
            # Put control/meta token early so model-side logic can switch system prompt.
            message.insert(control_insert_pos, dict(type='question_type', value='oe'))

        message.append(dict(type='text', value=prompt))

        

        return message

    def _build_mmmu_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MMMU dataset: keep all images at beginning."""

        import string

        import pandas as pd

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += 'Please select the correct answer from the options above. \n'
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_mcq_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for MCQ dataset: use chinese prompt if the question contains chinese characters."""
        MCQ_CN_PROMPT = '请直接回答选项字母。'
        MCQ_EN_PROMPT = 'Please select the correct answer from the options above.'

        import string

        import pandas as pd

        def cn_string(s):
            import re

            if re.search('[\u4e00-\u9fff]', s):
                return True
            return False

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        options = {cand: line[cand] for cand in string.ascii_uppercase if cand in line and not pd.isna(line[cand])}
        options_prompt = 'Options:\n'
        for key, item in options.items():
            options_prompt += f'{key}. {item}\n'
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        prompt = ''
        if hint is not None:
            prompt += f'Hint: {hint}\n'
        prompt += f'Question: {question}\n'
        if len(options):
            prompt += options_prompt
            prompt += MCQ_CN_PROMPT if cn_string(prompt) else MCQ_EN_PROMPT
        prompt = prompt.rstrip()
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        return msgs

    def _build_yorn_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for YORN dataset:"""
        YORN_PROMPT = ' Please answer yes or no.'

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += YORN_PROMPT
        return msgs

    def _build_vqa_prompt(self, line, dataset: str) -> list[dict[str, str]]:
        """change the prompt for VQA dataset:"""
        VQA_PROMPT = '\nPlease try to answer the question with short words or phrases if possible.'

        tgt_path = self.dump_image(line, dataset)
        question = line['question']
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=question))
        assert msgs[-1]['type'] == 'text'
        msgs[-1]['value'] += VQA_PROMPT
        return msgs
