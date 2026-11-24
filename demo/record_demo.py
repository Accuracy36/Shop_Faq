"""自动录制演示视频：启动 Demo 服务，用 Playwright 驱动无界面 Chromium 逐字输入问题并录屏，再转成 mp4。

    python demo/record_demo.py --model s2_final_roberta_seed1 --real-index --reject --output demo/video/demo.mp4

视频中的每一次问答都是服务端模型的实时推理结果；脚本只负责输入文字、点击和等待。
依赖：pip install playwright imageio-ffmpeg；python -m playwright install chromium
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (动作, 内容)：type 为逐字输入后回车；hot 为点击页面上方的热门问题；suggest 为点击“猜你想问”里的第一个
SCRIPT = [
    ('hot', '你们发什么快递？'),
    ('type', '我在重庆，大概几天能收到货啊'),
    ('type', '都三天了怎么还没给我发货！'),
    ('type', '买回来不喜欢能退吗'),
    ('type', '退的钱多久到账'),
    ('type', '发票可以开公司抬头吗'),
    ('type', '有没有优惠卷可以领'),
    ('type', '今天天气怎么样'),
    ('type', '这件衣服是纯棉的吗'),
    ('suggest', ''),
    ('type', '好的谢谢'),
]


def wait_for_server(port, process, timeout=90):
    start = time.time()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    last_error = ''
    while time.time() - start < timeout:
        if process.poll() is not None:
            raise RuntimeError(f'Demo服务退出：{process.returncode}；见server.log')
        try:
            with opener.open(f'http://127.0.0.1:{port}/api/info', timeout=2) as r:
                return json.loads(r.read().decode('utf-8'))
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1)
    raise RuntimeError('Demo服务启动超时：' + last_error)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--pseudo', default='')
    ap.add_argument('--eval-name', default='')
    ap.add_argument('--threshold', default='')
    ap.add_argument('--real-index', action='store_true')
    ap.add_argument('--reject', action='store_true')
    ap.add_argument('--port', type=int, default=7861)
    ap.add_argument('--output', default=str(ROOT / 'demo' / 'video' / 'demo.mp4'))
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    bundled_browser = ROOT / '.runtime' / 'playwright'
    if bundled_browser.exists():
        os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(bundled_browser))
    from playwright.sync_api import sync_playwright
    import imageio_ffmpeg

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = out.parent / 'raw'
    cmd = [sys.executable, str(ROOT / 'demo' / 'app.py'), '--model', args.model, '--port', str(args.port)]
    for flag, value in (('--pseudo', args.pseudo), ('--eval-name', args.eval_name), ('--threshold', args.threshold)):
        if value:
            cmd += [flag, value]
    cmd += [flag for flag, on in (('--cpu', args.cpu), ('--real-index', args.real_index), ('--reject', args.reject)) if on]
    server_log = out.with_suffix('.server.log').open('w', encoding='utf-8')
    server = subprocess.Popen(cmd, cwd=str(ROOT), stdout=server_log, stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    transcript = []
    try:
        info = wait_for_server(args.port, server)
        print('Demo ready', info, flush=True)
        with sync_playwright() as p:
            browser = p.chromium.launch()
            context = browser.new_context(viewport=dict(width=1280, height=720), device_scale_factor=1,
                                          record_video_dir=str(raw_dir), record_video_size=dict(width=1280, height=720))
            page = context.new_page()
            page.goto(f'http://127.0.0.1:{args.port}/')
            page.wait_for_selector('#hot .chip')
            page.wait_for_timeout(2500)
            for action, text in SCRIPT:
                before = page.locator('#chat .msg.bot').count()
                with page.expect_response('**/api/ask') as response:
                    if action == 'hot':
                        page.locator('#hot .chip', has_text=text).first.click()
                    elif action == 'suggest':
                        page.locator('#chat .msg.bot').last.locator('.chip').first.click()
                    else:
                        page.locator('#input').click()
                        page.keyboard.type(text, delay=90)
                        page.wait_for_timeout(350)
                        page.keyboard.press('Enter')
                reply = response.value.json()
                page.wait_for_function('n => document.querySelectorAll("#chat .msg.bot").length > n', arg=before)
                transcript.append(dict(action=action, input=text or reply['normalized'], status=reply['status'],
                                       answer=reply['answer'],
                                       matched=reply.get('question'), confidence=reply['confidence'],
                                       threshold=reply['threshold'], latency_ms=reply['latency_ms']))
                page.wait_for_timeout(3600)
            page.wait_for_timeout(1500)
            video = page.video
            context.close()
            webm = Path(video.path())
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=30)
        server_log.close()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', str(webm), '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-crf', '20', '-movflags', '+faststart', str(out)], check=True)
    (out.with_suffix('.json')).write_text(json.dumps(dict(model=info, transcript=transcript), ensure_ascii=False,
                                                     indent=2), encoding='utf-8')
    for row in transcript:
        print(row)
    print('视频：', out)


if __name__ == '__main__':
    main()
