"""Decode the complete MP4, validate transcript, and export inspection frames."""
import json
from pathlib import Path
import subprocess
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent.parent
VIDEO = ROOT / 'demo/video/PB24000178+周宇翔+简易购物自动问答系统.mp4'


def main():
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    stream = imageio_ffmpeg.read_frames(str(VIDEO))
    metadata = next(stream)
    stream.close()
    result = subprocess.run([ffmpeg, '-v', 'error', '-i', str(VIDEO), '-f', 'null', '-'],
                            capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
    transcript = json.loads(VIDEO.with_suffix('.json').read_text(encoding='utf-8'))
    checks = {
        'full_video_decodes': result.returncode == 0 and not result.stderr.strip(),
        'duration_10_to_120_seconds': 10 <= metadata['duration'] <= 120,
        'frame_size_1280_720': tuple(metadata['size']) == (1280, 720),
        'final_model': transcript['model']['model'] == 's2_final_roberta_seed1',
        'real_index_and_rejection': transcript['model']['index_size'] == 790 and transcript['model']['reject_size'] == 328,
        'normal_question_answered': transcript['transcript'][0]['status'] == 'answered',
        'paraphrase_answered': transcript['transcript'][1]['status'] == 'answered',
        'out_of_scope_rejected': all(r['status'] == 'rejected' for r in transcript['transcript'] if r['input'] in ['今天天气怎么样','这件衣服是纯棉的吗']),
        'eleven_live_responses': len(transcript['transcript']) == 11,
    }
    qa = VIDEO.parent / 'qa'
    qa.mkdir(exist_ok=True)
    frames = []
    for sec in [6, 14, 34, 50, 57, metadata['duration']-2]:
        frame = qa / f'frame_{sec:.1f}.png'
        subprocess.run([ffmpeg, '-v', 'error', '-y', '-ss', str(sec), '-i', str(VIDEO), '-frames:v', '1', str(frame)],
                       check=True, creationflags=subprocess.CREATE_NO_WINDOW)
        frames.append(str(frame))
    output = dict(checks=checks, metadata=metadata, frames=frames, decoder_exit=result.returncode)
    (ROOT / 'results/video_verification.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(output, ensure_ascii=False, indent=2))
    assert all(checks.values())


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    main()
