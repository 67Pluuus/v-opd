from huggingface_hub import snapshot_download
from ..smp import *
from ..smp.file import get_intermediate_file_path, get_file_extension
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE

FAIL_MSG = 'Failed to obtain answer via API.'


def unwrap_hf_pkl(pth, suffix='.mp4'):
    base_dir = os.path.join(pth, 'video_pkl/')
    target_dir = os.path.join(pth, 'video/')
    pickle_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    pickle_files.sort()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for pickle_file in pickle_files:
            with open(pickle_file, 'rb') as file:
                video_data = pickle.load(file)
            # For each video file in the pickle file, write its contents to a new mp4 file
            for video_name, video_content in video_data.items():
                output_path = os.path.join(target_dir, f'{video_name}{suffix}')
                with open(output_path, 'wb') as output_file:
                    output_file.write(video_content)
        print('The video file has been restored and stored from the pickle file.')
    else:
        print('The video file already exists.')


class Video_Holmes(VideoBaseDataset):

    MD5 = '85bdd91f9b29a99354c23b97ab7c113c'
    SYS = ''

    QUESTION_TMPL = """
    Based on the given video, reason and answer the single-choice question. Provide your reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags. \
    The question is: {}. The options are: {}. \
    Your answer:
    """  # noqa: E501

    TYPE = 'Video-MCQ'

    def __init__(self, dataset='Video_Holmes', nframe=-1, fps=2, frames_limit=768):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.dataset_name = dataset
        self.frames_limit = frames_limit

    @classmethod
    def supported_datasets(cls):
        return ['Video_Holmes']

    def prepare_dataset(self, dataset_name='Video_Holmes', json_path=None, repo_id=None):

        if repo_id is None:
            repo_id = os.getenv('VideoHolmes_PATH', None)
        if repo_id is None:
            raise ValueError('VideoHolmes_PATH is not set')
        
        dataset_path = repo_id

        if json_path is None:
            json_path = os.path.join(dataset_path, 'test_Video-Holmes.json')

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video_path']:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def unzip_hf_zip(pth):
                import zipfile
                base_dir = pth
                target_dir = os.path.join(pth, 'video/')
                zip_files = [
                    os.path.join(base_dir, file) for file in os.listdir(base_dir)
                    if file == "videos.zip"
                ]
                zip_files.sort()

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                    for zip_file in zip_files:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            for member in zip_ref.namelist():
                                # Check if the member is a file (not a directory)
                                if not member.endswith('/'):
                                    # Extract the file to the specified directory
                                    source = zip_ref.open(member)
                                    target = open(os.path.join(target_dir, os.path.basename(member)), 'wb')
                                    with source, target:
                                        target.write(source.read())
                    print('The video file has been restored and stored from the zip file.')
                else:
                    print('The video file already exists.')

            def generate_tsv(pth):

                data_file = osp.join(pth, f'{dataset_name}.tsv')
                
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return

                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                rows = []

                for idx, item in enumerate(data):

                    video_id = item.get('video ID')
                    options = item.get('Options', {})
                    candidates = [f"{k}. {options.get(k, '')}".replace("'","")
                                  for k in ['A', 'B', 'C', 'D', 'E', 'F'] if k in options]
                    row = {
                        'index': idx,
                        'video': video_id,
                        'video_path': f'./video/{video_id}.mp4',
                        'candidates': candidates,
                        'question': item.get('Question', ''),
                        'answer': item.get('Answer', ''),
                        'question_id': item.get('Question ID', ''),
                        'question_type': item.get('Question Type', ''),
                        'explanation': item.get('Explanation', ''),
                    }
                    rows.append(row)

                df = pd.DataFrame(rows)
                columns = ['index', 'video', 'video_path', 'candidates',
                           'question', 'answer', 'question_id', 'question_type', 'explanation']
                df = df[columns]
                if not os.path.exists(data_file):
                    df.to_csv(data_file, sep='\t', index=False)
                else:
                    print("The tsv file already exists.")

            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(data_file=data_file, root=dataset_path)

    def save_video_frames(self, video, video_llm=False, verbose=False):
        """
        Sample frames from <data_root>/video/<video>.mp4.
        Returns: (frame_paths, indices, video_info)
        """
        vid_path = osp.join(self.data_root, 'video', video + '.mp4')
        import decord
        vid = decord.VideoReader(vid_path)
        fps = float(vid.get_avg_fps())
        n_frames = int(len(vid))
        duration = (n_frames / fps) if fps > 0 else 0.0
        video_info = {
            'fps': fps,
            'n_frames': n_frames,
            'duration': duration,
        }

        if self.nframe > 0 and self.fps < 0:
            step_size = n_frames / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
        elif self.fps > 0:
            total_duration = duration
            required_frames = int(total_duration * self.fps) if total_duration > 0 else 0
            if required_frames > self.frames_limit:
                warnings.warn(
                    f"Video `{video}` requires {required_frames} frames at {self.fps} fps. "
                    f"Truncating to {self.frames_limit} frames."
                )
                video_info['sample_n_frames'] = self.frames_limit
                step_size = n_frames / (self.frames_limit + 1)
                indices = [int(i * step_size) for i in range(1, self.frames_limit + 1)]
                frame_root = osp.join(self.frame_root, video)
                os.makedirs(frame_root, exist_ok=True)
                frame_paths = [
                    osp.join(frame_root, self.frame_tmpl.format(i, self.frames_limit))
                    for i in range(1, self.frames_limit + 1)
                ]
                sample_fps = self.frames_limit / total_duration
            else:
                video_info['sample_n_frames'] = required_frames
                step_size = fps / self.fps if self.fps > 0 else 1.0
                indices = [int(i * step_size) for i in range(required_frames)]
                frame_paths = self.frame_paths_fps(video, len(indices))
                sample_fps = self.fps
            video_info['sample_fps'] = sample_fps

        else:
            raise ValueError('Either nframe > 0 or fps > 0 must be set.')

        if len(indices) == 0:
            # Degenerate case: extract at least one middle frame
            indices = [max(0, n_frames // 2)]
            frame_paths = self.frame_paths_fps(video, len(indices)) if self.fps > 0 else self.frame_paths(video)[:1]

        # clamp indices to valid range (avoid occasional out-of-range due to rounding)
        if n_frames > 0 and len(indices) > 0:
            max_idx = int(n_frames) - 1
            indices = [min(max(0, int(x)), max_idx) for x in indices]

        if not np.all([osp.exists(p) for p in frame_paths]):
            images = []
            for frame_idx in indices:
                images.append(Image.fromarray(vid[frame_idx].asnumpy()))
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths, indices, video_info

    def dump_prompt_info(self, line, use_frames=False):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video'], video_llm=True)

        info = {}

        info['video'] = osp.join(self.data_root, 'video', line['video'] + '.mp4')
        frames, indices, video_info = self.save_video_frames(line['video'], use_frames)
        
        info['raw_fps'] = video_info['fps']
        info['sample_fps'] = video_info['sample_fps']
        info['duration'] = video_info['duration']
        info['n_frames'] = video_info['n_frames']
        info['sample_n_frames'] = video_info['sample_n_frames']
        
        if use_frames:
            info['frames'] = frames

        info['question'] = line['question']
        info['candidates'] = '\n'.join(eval(line['candidates']))

        return info

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video'], video_llm=video_llm)

        # Start with system instruction
        message = [dict(type='text', value=self.SYS, role='system')]

        if video_llm:
            assert self.fps > 0
            actual_fps = (
                self.frames_limit / video_info['duration']
                if len(frames) == self.frames_limit and video_info['duration'] > 0
                else self.fps
            )
            message.append(dict(
                type='video',
                value=frames,
                sample_fps=actual_fps,
                min_pixels=1 * 2 * 2 * 16 * 16,
                max_pixels=768 * 32 * 32,
                total_pixels=224000 * 4 * 16 * 16,
            ))
        else:
            message.extend(dict(type='image', value=im) for im in frames)

        text_prompt = self.QUESTION_TMPL.format(line['question'], line['candidates'])
        message.append(dict(type='text', value=text_prompt))
        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):

        from .utils.videoholmes import get_dimension_rating, extract_option

        assert get_file_extension(eval_file) in ['xlsx', 'json', 'tsv'], 'data file should be an supported format (xlsx/json/tsv) file'  # noqa: E501

        tmp_file = get_intermediate_file_path(eval_file, '_tmp', 'pkl')
        tgt_file = get_intermediate_file_path(eval_file, '_rating', 'json')
        score_file = get_intermediate_file_path(eval_file, '_score')

        if 1:
            model = judge_kwargs.get('model', 'exact_matching')
            # assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']

            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
                model = None
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])

                predicted_answer = extract_option(pred)

                print(f"prediction: {pred}, predicted_answer: {predicted_answer}, ans: {ans}")

                data.loc[idx, 'score'] = int(predicted_answer == ans)

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating
