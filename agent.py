import os, json, time, subprocess, sys
from pathlib import Path
from typing import Optional
import requests
import asyncio
import edge_tts
import time
import jwt
from tqdm import tqdm
from sync import Sync as SyncSDK
from sync.common import Audio as SyncAudio, GenerationOptions as SyncGenOpt, Video as SyncVideo
from sync.core.api_error import ApiError as SyncApiError

ak = "AdAFbMfYKCdDnTknEkgd3G8ArdPeF4fE" # access key
sk = "frPbHdrGbdatrEek9LFRpbNpg4gd494k" # secret key

#kling 官方 jwt encoding
def encode_jwt_token(ak, sk):
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    payload = {
        "iss": ak,
        "exp": int(time.time()) + 1800, #
        "nbf": int(time.time()) - 5
    }
    token = jwt.encode(payload, sk, algorithm="HS256", headers=headers)
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token

API_KEY = encode_jwt_token(ak, sk)

# ================= 可灵API配置 ================
API_BASE = "https://api-beijing.klingai.com"  
KLING_MODEL = "kling-video/v1/standard/text-to-video" 
ASPECT_RATIO = "16:9"
DURATION_SEC = 5

WAV2LIP_DIR = Path("Wav2Lip")
WAV2LIP_CKPT = WAV2LIP_DIR / "checkpoints" / "Wav2Lip-SD-NOGAN.pt"  
WAV2LIP_INFER = WAV2LIP_DIR / "inference.py"

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

TTS_OUT = OUT_DIR / "tts.wav"
KLING_VIDEO = OUT_DIR / "kling_raw.mp4"
FINAL_OUT = OUT_DIR / "final_lipsync.mp4"

SYNC_API_KEY = "sk-86ckr6DeRZ6i6sVa3LlGZg.HPvfNG5g2I_SukS2CiD3pJObNyOcEZBT"

# ============ 步骤 1：LLM 文本 ============
# To be completed: Steps for the LLM insertion
def generate_script_with_llm(user_hint: str) -> dict:
    base_style = (
        "风格：综艺节目，写实，镜头运动自然；"
        "光线：柔和侧光；色彩：自然。"
    )
    voiceover = f"如今的教育界可算是出了大事。{user_hint}"
    kling_prompt = f"郭德纲在讲相声，边讲边做手势，衣着休闲，光线柔和，有轻微推拉和平移镜头。{base_style}"
    return {"voiceover": voiceover, "kling_prompt": kling_prompt}

# ============ 步骤 2：TTS edge-tts ============

async def synth_tts_to_wav(text: str, outfile: Path, voice: str = "zh-CN-YunjianNeural"):
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(outfile))

# ============ 步骤 3：Kling 可灵 文生视频 ===========
def create_kling_task(prompt: str, aspect_ratio="16:9", duration=DURATION_SEC) -> str:
    url = f"{API_BASE}/v1/videos/text2video"  # 官方API接口地址
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "product": "video",
        "version": "v1",
        "model": "text-to-video",
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "duration": duration
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json() if r.text else {}
    task_id = data.get("id") or data.get("request_id") or data.get("task_id")
    if not task_id:
        raise RuntimeError(f"未找到任务ID，返回内容：{data}")
    return task_id

def get_kling_result(task_id: str) -> dict:
    url = f"{API_BASE}//v1/videos/text2video/{task_id}"  # 官方API
    headers = {"Authorization": f"Bearer {API_KEY}"}
    params = {"id": task_id}
    r = requests.get(url, headers=headers, params=params, timeout=60)
    r.raise_for_status()
    return r.json() if r.text else {}

def parse_kling_result(resp: dict) -> str:
    data = resp.get('data') or resp
    if isinstance(data, dict):
        task_result = data.get('task_result', {})
        if isinstance(task_result, dict):
            videos = task_result.get('videos')
            if isinstance(videos, list) and videos:
                url = videos[0].get('url')
                if url and url.startswith('http'):
                    return url
            video_url = task_result.get('video_url')
            if video_url and video_url.startswith('http'):
                return video_url
    for path in [
        ["data", "video_url"],
        ["result", "video_url"],
        ["video_url"],
        ["url"]
    ]:
        cur = resp
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok and isinstance(cur, str) and cur.startswith("http"):
            return cur
    return None

def wait_and_download_kling(task_id: str, out_path: Path, timeout_sec=900, interval=5):
    """
    轮询直到成功或超时，然后下载视频到 out_path
    """
    import requests
    start = time.time()
    with tqdm(total=timeout_sec, desc="等待可灵生成", unit="s") as bar:
        while True:
            try:
                resp = get_kling_result(task_id)
                
            except requests.HTTPError as e:
                if e.response.status_code == 404:
                    print("404 error")
                    time.sleep(interval)
                    bar.update(interval)
                    continue
                else:
                    raise
            except Exception as e:
                print(f"获取可灵结果失败: {e}")
                time.sleep(interval)
                bar.update(interval)
                continue
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"Kling 超时，最后响应：{json.dumps(resp, ensure_ascii=False)[:500]}...")
            url = parse_kling_result(resp)
            if url:
                print(f"API返回：{resp}")
                print(f"[Kling] 视频就绪：{url}")
                with requests.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(out_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1<<20):
                            if chunk: f.write(chunk)
                print(f"[Kling] 下载完成: {out_path}")
                return
            time.sleep(interval)
            bar.update(interval)

# ============ 4：Wave2Lip 口型对齐 ============

def run_wav2lip(face_video: Path, audio_wav: Path, outfile: Path):
    if not WAV2LIP_CKPT.exists():
        raise FileNotFoundError(f"缺少权重：{WAV2LIP_CKPT}")
    cmd = [
        sys.executable, str(WAV2LIP_INFER),
        "--checkpoint_path", str(WAV2LIP_CKPT),
        "--face", str(face_video),
        "--audio", str(audio_wav),
        "--outfile", str(outfile),
        "--resize_factor", "2",
        "--face_det_batch_size", "1",
        "--wav2lip_batch_size", "1",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"  # 强制CPU
    subprocess.run(cmd, check=True, env=env)

def _temp_upload_to_tmpfile(local_path: Path) -> str:
    """
    将本地文件临时上传到 tmpfile 获取可公开访问的短期URL。
    """
    try:
        with open(local_path, "rb") as f:
            r2 = requests.post("https://tmpfiles.org/api/v1/upload",
                               files={"file": (local_path.name, f)}, timeout=180)
        r2.raise_for_status()
        data2 = r2.json()
        page_url = (data2.get("data") or {}).get("url")
        if not page_url:
            raise RuntimeError(f"tmpfiles 返回异常: {data2}")
        file_id = page_url.rstrip("/").split("/")[-1]
        return f"https://tmpfiles.org/dl/{file_id}"
    except Exception as e:
        raise RuntimeError(f"上传失败：file.io={getattr(r, 'status_code', 'NA')} {getattr(r, 'text', '')[:200]} | tmpfiles 异常: {e}")


def _upload_via_curl_0x0(local_path: Path) -> str:
    """使用 curl.exe 上传到 0x0.st，返回直链。失败抛异常。"""
    try:
        cmd = [
            "curl.exe", "-sS", "-F", f"file=@{str(local_path)}", "https://0x0.st"
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=False)
        url = out.decode("utf-8", errors="ignore").strip()
        if url.startswith("http"):
            return url
        raise RuntimeError(f"0x0.st 返回异常: {url[:200]}")
    except Exception as e:
        raise RuntimeError(f"curl.exe 上传 0x0.st 失败: {e}")


def _upload_via_requests_0x0(local_path: Path) -> str:
    """使用 requests 上传到 0x0.st,返回直链。"""
    with open(local_path, "rb") as f:
        r = requests.post("https://0x0.st", files={"file": (local_path.name, f)}, timeout=300)
    r.raise_for_status()
    url = (r.text or "").strip()
    if url.startswith("http"):
        return url
    raise RuntimeError(f"0x0.st 返回异常: {r.status_code} {r.text[:200]}")


def _upload_via_0x0(local_path: Path) -> str:
    """优先用 curl.exe，失败回退 requests。"""
    try:
        return _upload_via_curl_0x0(local_path)
    except Exception as e:
        print(f"[upload] curl.exe 0x0.st 失败，回退 requests：{e}")
        return _upload_via_requests_0x0(local_path)

def run_sync_lipsync(face_video: Path, audio_wav: Path, outfile: Path,
                      model: str = "lipsync-2", sync_mode: str = "cut_off"):
    """
    使用 Sync.so 云端口型对齐API：
    """
    api_key = SYNC_API_KEY

    if not face_video.exists():
        raise FileNotFoundError(f"找不到本地视频：{face_video}")
    if not audio_wav.exists():
        raise FileNotFoundError(f"找不到本地音频/视频：{audio_wav}")

    proc_video = face_video
    proc_audio = audio_wav

    # 仅使用 0x0.st 上传，返回直链
    print("[Sync] 上传原始视频到 0x0.st ...")
    video_url = _upload_via_0x0(proc_video)
    print(f"[Sync] 视频URL: {video_url}")
    print("[Sync] 上传原始音频到 0x0.st ...")
    audio_url = _upload_via_0x0(proc_audio)
    print(f"[Sync] 音频URL: {audio_url}")

    print("[Sync] 创建任务")
    base_url = "https://api.sync.so"
    client = SyncSDK(base_url=base_url, api_key=api_key).generations
    try:
        resp = client.create(
            input=[SyncVideo(url=video_url), SyncAudio(url=audio_url)],
            model=model,
            options=SyncGenOpt(sync_mode=sync_mode)
        )
    except SyncApiError as e:
        raise RuntimeError(f"[Sync SDK] 创建任务失败: {e.status_code} {e.body}")

    job_id = resp.id
    print(f"[Sync] job_id = {job_id}")

    start = time.time()
    while True:
        generation = client.get(job_id)
        status = generation.status
        if status in ("COMPLETED", "FAILED", "REJECTED"):
            print(f"[Sync] 任务状态：{status}")
            if status in ("FAILED", "REJECTED"):
                reason = getattr(generation, "error", None) or getattr(generation, "message", None)
                raise RuntimeError(f"[Sync SDK] 任务失败：status={status}, reason={reason or generation}")
            output_url = getattr(generation, "output_url", None)
            if not output_url:
                raise RuntimeError(f"[Sync SDK] 未找到输出URL: {generation}")
            break
        print(f"[Sync] 等待中（状态：{status}）...")
        time.sleep(10)
        if time.time() - start > 1800:
            raise TimeoutError("[Sync] 口型任务超时")

    with requests.get(output_url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(outfile, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
    print(f"[Sync] 下载完成：{outfile}")


def main(user_hint: str):
    # 1) 文案
    plan = generate_script_with_llm(user_hint)
    print("\n[脚本-配音稿]\n", plan["voiceover"])
    print("\n[可灵提示词]\n", plan["kling_prompt"])

    # 2) 语音
    print("\n[2] 生成配音 ...")
    asyncio.run(synth_tts_to_wav(plan["voiceover"], TTS_OUT))

    # 3) 文生视频
    print("\n[3] 提交可灵任务 ...")
    task_id = create_kling_task(prompt=plan["kling_prompt"], aspect_ratio=ASPECT_RATIO, duration=DURATION_SEC)
    print(f"[Kling] task_id = {task_id}")
    wait_and_download_kling(task_id, KLING_VIDEO)

    # 4) 口型对齐
    print("\n[4] wave to lip online...")
    run_sync_lipsync(KLING_VIDEO, TTS_OUT, FINAL_OUT)
    print(f"\n 完成！{FINAL_OUT.resolve()}")
    return
    # print("\n[4] 本地 Wav2Lip 口型对齐 ...")
    # run_wav2lip(KLING_VIDEO, TTS_OUT, FINAL_OUT)
    # print(f"\n 完成！成品视频：{FINAL_OUT.resolve()}")

if __name__ == "__main__":
    user_hint = "从创意到成片的自动化agent演示。"
    main(user_hint)
