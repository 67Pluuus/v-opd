# build_tiny_from_existing_videos.py
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        default="../dataset/Seeker-173K/SFT/sft_llava-video_youtube_qa_mc_2_3_m_clue_multi_w_tool_diff_2790.json",
        help="Seeker annotation json path",
    )
    parser.add_argument(
        "--video-dir",
        default="../dataset/LLaVA-Video-178K/2_3_m_youtube_v0_1/liwei_youtube_videos/videos/youtube_video_2024",
        help="Folder containing downloaded videos",
    )
    parser.add_argument(
        "--output-json",
        default="../dataset/Seeker-173K/SFT/tiny.json",
        help="Output annotation json path",
    )
    args = parser.parse_args()

    input_json = Path(args.input_json)
    video_dir = Path(args.video_dir)
    output_json = Path(args.output_json)

    data = load_json(input_json)

    kept = []
    missing = []

    for sample in data:
        videos = sample.get("videos", [])
        if not videos:
            continue

        first_url = videos[0].get("url")
        if not first_url:
            continue

        video_name = Path(first_url).name
        video_path = video_dir / video_name

        if video_path.exists():
            kept.append(sample)
        else:
            missing.append(video_name)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"Total samples: {len(data)}")
    print(f"Kept samples: {len(kept)}")
    print(f"Missing videos: {len(missing)}")
    print(f"Saved to: {output_json}")


if __name__ == "__main__":
    main()