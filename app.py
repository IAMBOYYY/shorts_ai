import asyncio, os, uuid, json, subprocess, random, re, textwrap, math
import httpx, edge_tts, numpy as np, soundfile as sf
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, BackgroundTasks, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List
from urllib.parse import quote

app = FastAPI()

OR_KEYS = [os.environ.get(f"OR_KEY_{i}", "") for i in range(1, 4)]
GROQ_KEYS = [os.environ.get(f"GROQ_KEY_{i}", "") for i in range(1, 4)]
# OpenRouter fallback models
FREE_MODELS = [
    "meta-llama/llama-4-scout:free",
    "mistralai/mistral-7b-instruct:free",
    "openrouter/cypher-alpha:free",
    "openrouter/free",
]
# Groq models (tried first - free, fast, high rate limits)
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
]
OUTPUT_DIR = Path("/tmp/shortsai")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
jobs: dict = {}

W9, H9 = 1080, 1920   # 9:16
W16, H16 = 1920, 1080  # 16:9

# ─── FONTS ────────────────────────────────────────────────────────
def get_font(size=36, bold=False):
    paths = [
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
        f"/usr/share/fonts/truetype/liberation/LiberationSans{'-Bold' if bold else '-Regular'}.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in paths:
        if Path(p).exists():
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()

# ─── REQUEST ──────────────────────────────────────────────────────
class VideoRequest(BaseModel):
    template: str = "cinematic"        # cinematic | iphone_chat | instagram_dm | reddit_story | quiz
    title: str = ""
    niche: str = "motivation"
    tone: str = "energetic"
    script_mode: str = "ai"
    script_text: str = ""
    topic: str = ""
    voice: str = "male"
    voice2: str = "female"             # second voice for chat templates
    voice_speed: float = 1.0
    duration: int = 60
    aspect_ratio: str = "9:16"
    video_style: str = "fast_cuts"
    bg_music: str = "ambient"
    use_stock: bool = True
    session_id: str = ""
    # Chat template specific
    person1_name: str = "Alex"
    person2_name: str = "Jordan"
    # Reddit specific
    subreddit: str = "AITA"
    reddit_username: str = "throwaway_user"
    # Quiz specific
    num_questions: int = 5

# ─── OPENROUTER ───────────────────────────────────────────────────
async def call_openrouter(prompt: str, system: str = "") -> str:
    last = "No keys"
    msgs = [{"role":"system","content":system},{"role":"user","content":prompt}]

    # ── Try Groq first (free, fast, generous rate limits) ──────────
    for ki, key in enumerate(GROQ_KEYS):
        if not key.strip(): continue
        for model in GROQ_MODELS:
            try:
                async with httpx.AsyncClient(timeout=60) as c:
                    r = await c.post("https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key.strip()}",
                                 "Content-Type": "application/json"},
                        json={"model": model, "messages": msgs,
                              "max_tokens": 4096, "temperature": 0.7})
                    if r.status_code == 200:
                        txt = r.json()["choices"][0]["message"]["content"]
                        if txt and txt.strip():
                            print(f"✅ Groq Key{ki+1} {model}")
                            return txt
                    elif r.status_code == 429:
                        last = f"Groq Key{ki+1}/{model}: 429"; break
                    else:
                        last = f"Groq Key{ki+1}/{model}: {r.status_code}"
            except Exception as e:
                last = f"Groq error: {e}"

    # ── Fallback: OpenRouter ────────────────────────────────────────
    for ki, key in enumerate(OR_KEYS):
        if not key.strip(): continue
        for model in FREE_MODELS:
            try:
                async with httpx.AsyncClient(timeout=90) as c:
                    r = await c.post("https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {key.strip()}",
                                 "Content-Type": "application/json",
                                 "HTTP-Referer": "https://shortsai.hf.space",
                                 "X-Title": "ShortsAI"},
                        json={"model": model, "messages": msgs,
                              "max_tokens": 4096, "temperature": 0.7})
                    if r.status_code == 200:
                        txt = r.json()["choices"][0]["message"]["content"]
                        if txt and txt.strip():
                            print(f"✅ OR Key{ki+1} {model}")
                            return txt
                        last = f"OR Key{ki+1}/{model}: empty"
                    else:
                        last = f"OR Key{ki+1}/{model}: {r.status_code}"
                        if r.status_code == 429: await asyncio.sleep(8); break
            except Exception as e:
                last = str(e)
    raise Exception(f"All keys failed: {last}")

def clean_json_array(raw: str) -> str:
    """Like clean_json but extracts [ arrays first (for chat/list responses)."""
    if not raw or not raw.strip():
        raise Exception("AI returned empty response")
    raw = raw.strip()
    for fence in ["```json","```"]:
        if fence in raw:
            parts = raw.split(fence)
            raw = parts[1] if len(parts)>1 else parts[0]
            if "```" in raw: raw = raw.split("```")[0]
            break
    raw = raw.strip()
    # For arrays: extract [ first, then { fallback
    for sc, ec in [("[","]"),("{","}")]:
        si, ei = raw.find(sc), raw.rfind(ec)
        if si != -1 and ei > si:
            raw = raw[si:ei+1]; break
    return _repair_json(raw) if raw else raw

def clean_json(raw: str) -> str:
    if not raw or not raw.strip():
        raise Exception("AI returned empty response — try again")
    raw = raw.strip()
    # strip markdown fences
    for fence in ["```json","```"]:
        if fence in raw:
            parts = raw.split(fence)
            raw = parts[1] if len(parts)>1 else parts[0]
            if "```" in raw: raw = raw.split("```")[0]
            break
    raw = raw.strip()
    # extract JSON block even if model adds prose
    for sc, ec in [("{","}"),("[","]")]:
        si, ei = raw.find(sc), raw.rfind(ec)
        if si != -1 and ei > si:
            raw = raw[si:ei+1]; break
    if not raw:
        raise Exception("AI returned no parseable JSON — try again")
    return _repair_json(raw)

def _repair_json(s: str) -> str:
    """Best-effort repair of truncated/malformed JSON."""
    import json
    try:
        json.loads(s); return s  # already valid
    except Exception:
        pass
    # Remove trailing commas before ] or }
    import re
    s = re.sub(r",\s*([\]}])", r"\1", s)
    try:
        json.loads(s); return s
    except Exception:
        pass
    # Try to close unclosed brackets/braces
    stack = []
    in_str = False
    escape = False
    for ch in s:
        if escape: escape = False; continue
        if ch == "\\": escape = True; continue
        if ch == '"' and not escape: in_str = not in_str; continue
        if not in_str:
            if ch in "{[": stack.append("}" if ch=="{" else "]")
            elif ch in "}]" and stack: stack.pop()
    s = s + "".join(reversed(stack))
    try:
        json.loads(s); return s
    except Exception:
        pass
    # Last resort: truncate to last complete value
    for i in range(len(s)-1, 0, -1):
        if s[i] in ",":
            candidate = s[:i] + "".join(reversed(stack if stack else []))
            try:
                json.loads(candidate); return candidate
            except Exception:
                continue
    return s  # return as-is, let caller handle

# ─── TTS ──────────────────────────────────────────────────────────
VOICES = {
    # Male voices
    "male":"en-US-GuyNeural",
    "deep":"en-US-ChristopherNeural",
    "energetic":"en-US-BrandonNeural",
    "news":"en-US-SteffanNeural",
    "british_male":"en-GB-RyanNeural",
    "australian_male":"en-AU-WilliamNeural",
    "indian_male":"en-IN-PrabhatNeural",
    "casual_male":"en-US-EricNeural",
    "young_male":"en-US-AndrewNeural",
    # Female voices
    "female":"en-US-JennyNeural",
    "soft":"en-US-AriaNeural",
    "whispery":"en-US-AnaNeural",
    "british_female":"en-GB-SoniaNeural",
    "australian_female":"en-AU-NatashaNeural",
    "indian_female":"en-IN-NeerjaNeural",
    "warm_female":"en-US-MichelleNeural",
    "young_female":"en-US-EmmaNeural",
    "narrator_female":"en-US-CoraNeural",
    # Special
    "british":"en-GB-RyanNeural",
    "australian":"en-AU-WilliamNeural",
    "indian":"en-IN-PrabhatNeural",
}
async def tts(text: str, voice: str, path: Path, speed: float = 1.0):
    v = VOICES.get(voice, "en-US-GuyNeural")
    rate = int((speed-1.0)*100)
    rate_s = f"+{rate}%" if rate >= 0 else f"{rate}%"
    await edge_tts.Communicate(text, v, rate=rate_s).save(str(path))

def audio_dur(path: Path) -> float:
    r = subprocess.run(["ffprobe","-v","quiet","-print_format","json",
                        "-show_streams",str(path)], capture_output=True, text=True)
    try:
        for s in json.loads(r.stdout).get("streams",[]):
            if s.get("codec_type")=="audio": return float(s.get("duration",30))
    except: pass
    return 30.0

# ─── BACKGROUND MUSIC ─────────────────────────────────────────────
def gen_music(dur: float, mood: str, path: Path, sr=44100):
    if mood=="none": return False
    n = int(sr*dur); t = np.linspace(0,dur,n,endpoint=False)
    a = np.zeros(n,dtype=np.float32)
    if mood=="ambient":
        lfo=0.5+0.5*np.sin(2*np.pi*0.08*t)
        a=(np.sin(2*np.pi*130.8*t)*0.28+np.sin(2*np.pi*196*t)*0.16+np.sin(2*np.pi*261.6*t)*0.10)*lfo
    elif mood=="dramatic":
        lfo=0.4+0.6*np.abs(np.sin(2*np.pi*0.06*t))
        a=(np.sin(2*np.pi*55*t)*0.35+np.sin(2*np.pi*82.4*t)*0.22+np.sin(2*np.pi*58.3*t)*0.08)*lfo
    elif mood=="upbeat":
        pulse=(np.sin(2*np.pi*2.0*t)>0).astype(np.float32)
        a=(np.sin(2*np.pi*220*t)*0.22+np.sin(2*np.pi*330*t)*0.12)*(0.5+0.5*pulse)+np.sin(2*np.pi*110*t)*0.12
    elif mood=="dark":
        lfo=0.3+0.7*np.abs(np.sin(2*np.pi*0.04*t))
        a=(np.sin(2*np.pi*40*t)*0.30+np.sin(2*np.pi*60*t)*0.18+np.random.randn(n).astype(np.float32)*0.025)*lfo
    elif mood=="tense":
        tr=0.5+0.5*np.sin(2*np.pi*7*t)
        a=np.sin(2*np.pi*73.4*t)*0.22*tr+np.sin(2*np.pi*77.8*t)*0.08+np.sin(2*np.pi*146.8*t)*0.12
    fade=min(int(sr*2),n//5)
    a[:fade]*=np.linspace(0,1,fade,dtype=np.float32)
    a[-fade:]*=np.linspace(1,0,fade,dtype=np.float32)
    pk=np.max(np.abs(a))
    if pk>0: a=np.clip(a/pk*0.5,-1,1)
    sf.write(str(path),a,sr); return True

# ─── IMAGE HELPERS ────────────────────────────────────────────────
NICHE_STYLES = {
    "tech":"futuristic neon holographic UI, cyberpunk cityscape, blue cyan tones, cinematic 8K",
    "finance":"luxury Wall Street office, golden coins and charts, dramatic professional lighting",
    "horror":"dark eerie abandoned location, fog, horror film grain, ominous shadows",
    "motivation":"epic mountain peak silhouette, golden hour sunrise, vast dramatic sky",
    "facts":"colorful vibrant educational illustration, bold graphic design, clean modern style",
    "history":"vintage documentary sepia photograph, period-accurate historical setting",
    "science":"macro laboratory photography, glowing molecular structures, clean scientific lighting",
    "gaming":"epic cinematic game concept art, neon RGB, dramatic hero shot",
    "nature":"breathtaking wildlife photography, golden hour, National Geographic style",
    "food":"gourmet food close-up, steam rising, warm bokeh kitchen, Michelin star",
    "crypto":"bitcoin digital gold coins, blockchain visualization, neon dark background",
    "space":"NASA quality nebula and planets, deep space 8K, cosmic dramatic lighting",
    "psychology":"surreal dreamlike mindscape, symbolic lighting, conceptual art",
    "fitness":"athletic peak performance, cinematic gym lighting, raw energy",
    "travel":"breathtaking travel destination, golden hour, wide angle cinematic",
    "celebrity":"glamorous luxury lifestyle, magazine quality portrait",
    "drama":"cinematic emotional close-up, film noir shadows, dramatic lighting",
    "ragebait":"shocking dramatic scene, bold red tones, high contrast provocative imagery",
    "conspiracy":"dark evidence board, dim lighting, mysterious shadows",
    "comedy":"bright pop art comic book style, funny exaggerated, bold colors",
    "animals":"adorable wildlife close-up portrait, soft natural light",
}
NICHE_KW = {"tech":"technology futuristic","finance":"business money","horror":"dark eerie",
            "motivation":"motivation success","facts":"education knowledge","history":"history ancient",
            "science":"science laboratory","gaming":"gaming neon","nature":"nature landscape",
            "food":"food gourmet","crypto":"cryptocurrency bitcoin","space":"space galaxy",
            "psychology":"mind psychology","fitness":"fitness gym","travel":"travel adventure",
            "celebrity":"luxury glamour","drama":"cinematic dramatic","ragebait":"shocking dramatic",
            "conspiracy":"dark mysterious","comedy":"funny colorful","animals":"animals wildlife"}
NICHE_COLOR = {
    "horror":"eq=saturation=0.35:contrast=1.45:brightness=-0.06",
    "motivation":"eq=saturation=1.6:contrast=1.1:brightness=0.05",
    "drama":"eq=saturation=0.75:contrast=1.35","conspiracy":"eq=saturation=0.25:contrast=1.55:brightness=-0.08",
    "ragebait":"eq=saturation=1.9:contrast=1.35","space":"eq=saturation=1.4:contrast=1.2:brightness=-0.04",
    "finance":"eq=saturation=1.15:contrast=1.1","tech":"eq=saturation=1.2:contrast=1.15",
    "nature":"eq=saturation=1.4:contrast=1.05","fitness":"eq=saturation=1.5:contrast=1.25",
    "food":"eq=saturation=1.5:contrast=1.1","default":"eq=saturation=1.1:contrast=1.05",
}

async def _try_pollinations(prompt: str, path: Path, W: int, H: int) -> bool:
    """Try Pollinations - free, no auth required. Optional key for higher rate limits."""
    cw, ch = min(W, 1024), min(H, 1024)
    short_prompt = prompt[:500]
    POLL_KEY = os.environ.get("POLLINATIONS_KEY", "")

    # ── Method 1: GET endpoint — gen.pollinations.ai (new unified URL, no key needed) ──
    seed = uuid.uuid4().int % 99999
    for model in ["flux", "kontext", "seedream"]:
        params = f"model={model}&width={cw}&height={ch}&seed={seed}"
        if POLL_KEY:
            params += f"&key={POLL_KEY}"
        url = f"https://gen.pollinations.ai/image/{quote(short_prompt)}?{params}"
        try:
            async with httpx.AsyncClient(timeout=90, follow_redirects=True) as c:
                r = await c.get(url)
                if r.status_code == 200 and len(r.content) > 5000:
                    path.write_bytes(r.content)
                    print(f"✅ Pollinations GET {model}")
                    return True
                elif r.status_code == 429:
                    await asyncio.sleep(10)
                else:
                    print(f"Pollinations GET {model}: {r.status_code}")
        except Exception as e:
            print(f"Pollinations GET {model} error: {e}")

    # ── Method 2: OpenAI-compatible POST (more reliable with API key) ──────────────
    if POLL_KEY:
        for model in ["flux", "kontext"]:
            try:
                async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
                    r = await c.post(
                        "https://gen.pollinations.ai/v1/images/generations",
                        headers={"Authorization": f"Bearer {POLL_KEY}",
                                 "Content-Type": "application/json"},
                        json={"model": model, "prompt": short_prompt,
                              "n": 1, "size": f"{cw}x{ch}", "response_format": "url"}
                    )
                    if r.status_code == 200:
                        img_url = r.json().get("data", [{}])[0].get("url", "")
                        if img_url:
                            r2 = await c.get(img_url)
                            if r2.status_code == 200 and len(r2.content) > 5000:
                                path.write_bytes(r2.content)
                                print(f"✅ Pollinations POST {model}")
                                return True
                    elif r.status_code == 429:
                        await asyncio.sleep(8)
                    else:
                        print(f"Pollinations POST {model}: {r.status_code}")
            except Exception as e:
                print(f"Pollinations POST {model} error: {e}")

    return False

async def _try_hf_image(prompt: str, path: Path, W: int, H: int) -> bool:
    """HuggingFace Inference API - free fallback for image generation."""
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    capped_w, capped_h = min(W, 1024), min(H, 1024)
    models = [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/stable-diffusion-xl-base-1.0",
    ]
    for model_id in models:
        try:
            async with httpx.AsyncClient(timeout=120) as c:
                r = await c.post(
                    f"https://api-inference.huggingface.co/models/{model_id}",
                    headers=headers,
                    json={"inputs": prompt[:500], "parameters": {"width": capped_w, "height": capped_h}},
                )
                if r.status_code == 200 and len(r.content) > 5000:
                    path.write_bytes(r.content); return True
                elif r.status_code == 503:  # model loading
                    await asyncio.sleep(20)
        except Exception as e:
            print(f"HF image error {model_id}: {e}")
    return False

async def download_image(prompt: str, path: Path, W: int, H: int):
    """Try multiple free image sources in order of reliability."""
    # 1. Pollinations
    if await _try_pollinations(prompt, path, W, H): return

    # 2. HF Inference (only if token set, since it fails with DNS otherwise)
    if os.environ.get("HF_TOKEN") and await _try_hf_image(prompt, path, W, H): return

    # 3. Unsplash — smarter keyword extraction from scene prompt
    # Take the most descriptive nouns from the prompt (skip style words)
    skip_words = {"photorealistic","cinematic","8k","4k","shot","ultra","detailed",
                  "lighting","angle","style","color","palette","camera","high","quality"}
    kw_list = [w.lower() for w in prompt.replace(","," ").split()
               if len(w) > 3 and w.lower() not in skip_words][:4]
    keywords = "+".join(kw_list) if kw_list else "nature"
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=25, follow_redirects=True) as c:
                r = await c.get(f"https://source.unsplash.com/{W}x{H}/?{keywords}")
                if r.status_code == 200 and len(r.content) > 5000:
                    path.write_bytes(r.content)
                    print(f"✅ Unsplash: {keywords[:40]}")
                    return
        except Exception as e:
            print(f"Unsplash attempt {attempt+1}: {e}")
            await asyncio.sleep(2)

    # 4. Picsum — random beautiful real photo, always works
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
            r = await c.get(f"https://picsum.photos/{min(W,1080)}/{min(H,1920)}")
            if r.status_code == 200 and len(r.content) > 5000:
                path.write_bytes(r.content)
                print("✅ Picsum fallback image")
                return
    except Exception as e:
        print(f"Picsum error: {e}")

    # 5. Absolute last resort: dark gradient PIL image
    img = Image.new("RGB", (W, H), color=(15, 15, 30))
    img.save(path, "JPEG", quality=85)
    print("⚠️ Using PIL placeholder — all image sources failed")

async def fetch_pexels_video(query: str, path: Path) -> bool:
    """Fetch a free stock video clip from Pexels (requires PEXELS_KEY env var)."""
    PEXELS_KEY = os.environ.get("PEXELS_KEY", "")
    if not PEXELS_KEY: return False
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as c:
            r = await c.get(
                f"https://api.pexels.com/videos/search?query={quote(query)}&per_page=8&orientation=portrait",
                headers={"Authorization": PEXELS_KEY}
            )
            if r.status_code != 200: return False
            videos = r.json().get("videos", [])
            if not videos: return False
            vid = random.choice(videos[:4])
            files = vid.get("video_files", [])
            # prefer HD portrait
            files_sorted = sorted(files, key=lambda f: f.get("width", 0) or 0)
            pick = next((f for f in files_sorted if 400 <= (f.get("width") or 0) <= 1080), files_sorted[-1] if files_sorted else None)
            if not pick: return False
            vr = await c.get(pick["link"], follow_redirects=True)
            if vr.status_code == 200 and len(vr.content) > 10000:
                path.write_bytes(vr.content); return True
    except Exception as e:
        print(f"Pexels video error: {e}")
    return False

async def fetch_stock(kw: str, path: Path, W: int, H: int) -> bool:
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as c:
            r = await c.get(f"https://source.unsplash.com/{W}x{H}/?{quote(kw)}")
            if r.status_code==200 and len(r.content)>5000: path.write_bytes(r.content); return True
    except: pass
    return False

def create_srt(script, dur, path):
    words = script.split(); chunk = 6
    chunks = [" ".join(words[i:i+chunk]) for i in range(0,len(words),chunk)]
    if not chunks: return
    cd = dur/len(chunks)
    def ft(t):
        h,m,s,ms=int(t//3600),int((t%3600)//60),int(t%60),int((t*1000)%1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    path.write_text("\n".join(f"{i+1}\n{ft(i*cd)} --> {ft((i+1)*cd)}\n{c}\n" for i,c in enumerate(chunks)))

def build_image_clip(img: Path, out: Path, W, H, dur, crop_idx, color_filter):
    ox,oy=[(0,0),(0.12,0),(0,0.12),(0.06,0.06),(0.12,0.12),(0.06,0)][crop_idx%6]
    sw,sh=int(W*1.15),int(H*1.15)
    vf=(f"scale={sw}:{sh}:force_original_aspect_ratio=increase,"
        f"crop={W}:{H}:x={int(ox*W)}:y={int(oy*H)},{color_filter}")
    r=subprocess.run(["ffmpeg","-y","-loop","1","-i",str(img),"-t",str(dur),
                       "-vf",vf,"-r","25","-c:v","libx264","-pix_fmt","yuv420p",
                       "-preset","ultrafast",str(out)],capture_output=True)
    if r.returncode!=0: raise Exception(f"Clip: {r.stderr.decode()[-300:]}")

def concat_clips(paths, out):
    valid=[p for p in paths if Path(p).exists() and Path(p).stat().st_size>500]
    if not valid: raise Exception("No valid clips to concat")
    if len(valid)<len(paths): print(f"⚠️ Skipping {len(paths)-len(valid)} missing clips")
    lf=out.parent/"list.txt"
    lf.write_text("\n".join(f"file '{p}'" for p in valid))
    r=subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(lf),"-c","copy",str(out)],capture_output=True)
    if r.returncode!=0:
        r2=subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",str(lf),
                           "-c:v","libx264","-c:a","aac","-preset","ultrafast",str(out)],capture_output=True)
        if r2.returncode!=0: raise Exception(f"Concat: {r2.stderr.decode()[-300:]}")

def merge_audio(video, voice, music, srt, output, has_music):
    srt_esc=str(srt).replace("\\","/").replace(":","\\:")
    sub_filter=(f"subtitles={srt_esc}:force_style='"
        "FontName=DejaVu Sans Bold,FontSize=18,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H00000000,BackColour=&H80000000,Outline=2,Shadow=1,"
        "Bold=1,Alignment=2,MarginV=50'")
    if has_music and music and music.exists():
        cmd=["ffmpeg","-y","-i",str(video),"-i",str(voice),"-i",str(music),
             "-filter_complex","[1:a]volume=1.0[v];[2:a]volume=0.18[m];[v][m]amix=inputs=2:duration=first[aout]",
             "-map","0:v","-map","[aout]","-vf",sub_filter,
             "-c:v","libx264","-c:a","aac","-b:a","128k","-shortest","-preset","ultrafast",str(output)]
    else:
        cmd=["ffmpeg","-y","-i",str(video),"-i",str(voice),"-vf",sub_filter,
             "-c:v","libx264","-c:a","aac","-b:a","128k","-shortest","-preset","ultrafast",str(output)]
    r=subprocess.run(cmd,capture_output=True)
    if r.returncode!=0:
        cmd2=(cmd[:cmd.index("-vf")]+cmd[cmd.index("-vf")+2:]) if "-vf" in cmd else cmd
        cmd2=[x for x in cmd2 if x!=sub_filter]
        r2=subprocess.run(cmd2,capture_output=True)
        if r2.returncode!=0: raise Exception(f"Merge: {r2.stderr.decode()[-300:]}")

def get_cut_times(dur, style):
    times=[0.0]; t=0.0
    intervals={"fast_cuts":(0.4,0.9),"cinematic":(1.4,2.8),"documentary":(3.0,5.5)}
    lo,hi=intervals.get(style,(1.5,3.0))
    while t<dur-0.3:
        t+=random.uniform(lo,hi)
        if t<dur: times.append(round(t,2))
    return times

# ══════════════════════════════════════════════════════════════════
# TEMPLATE 1: CINEMATIC (existing enhanced)
# ══════════════════════════════════════════════════════════════════
async def render_cinematic(job_id, req, job_dir, upd):
    W,H=(W9,H9) if req.aspect_ratio=="9:16" else (W16,H16)
    num_scenes=4 if req.duration<=30 else 6
    img_style=NICHE_STYLES.get(req.niche,"cinematic photorealistic 8K")
    color_filter=NICHE_COLOR.get(req.niche,NICHE_COLOR["default"])

    upd(5,"✍️ AI writing viral script...")
    if req.script_mode=="ai":
        wps=2.3*req.voice_speed; tw=int(req.duration*wps)
        prompt=f"""You are a professional YouTube Shorts scriptwriter. Write a VIRAL {req.duration}-second voiceover script.

TOPIC: {req.topic}
NICHE: {req.niche} | TONE: {req.tone} | SCENES: {num_scenes}

SCRIPT RULES:
- EXACTLY {tw} words (count every word — this is critical)
- Start with a POWERFUL hook that stops the scroll (question, shocking fact, or bold claim)
- Use short punchy sentences. Build tension. End with a revelation or punchline.
- Pure spoken words ONLY — no stage directions, brackets, or timestamps
- Make it emotional, specific, and impossible to skip

SCENE IMAGE PROMPT RULES:
- Each scene: describe EXACTLY what the camera sees (not feelings or abstract ideas)
- Include: subject + action + environment + lighting + camera angle
- Example good prompt: "young woman crying alone at night, rain on window, close-up wet face, blue neon reflection, cinematic 4K"
- Match scene to the script moment — scene 1 = hook moment, last scene = payoff

Return ONLY this JSON (no other text):
{{"script":"EXACTLY {tw} words here","scenes":[{{"description":"visual camera prompt here","index":0}}]}}"""
        raw=clean_json(await call_openrouter(prompt,"Return ONLY valid JSON, no markdown."))
        data=json.loads(raw)
        # Handle if AI returned array instead of dict
        if isinstance(data, list): data={"script": " ".join(str(d) for d in data), "scenes":[]}
        script=re.sub(r'\[[\d:]+\]\s*|Scene\s+\d+\s*:?|\[\s*Scene\s+\d+\s*\]','',data.get("script",""),flags=re.I).strip()
        raw_scenes=data.get("scenes",[])[:num_scenes]
    else:
        script=req.script_text.strip()
        raw2=clean_json(await call_openrouter(
            f"Split into {num_scenes} visual scenes.\nScript:{script}\nReturn JSON array:[{{\"description\":\"30-50 word image prompt\",\"index\":0}}]",
            "Return ONLY valid JSON array."))
        parsed=json.loads(raw2); raw_scenes=(parsed if isinstance(parsed,list) else parsed.get("scenes",[]))[:num_scenes]

    # Script length check + retry
    tw_target=int(req.duration*2.3*req.voice_speed)
    if len(script.split())<tw_target*0.65:
        upd(12,"⚠️ Script too short, AI retrying...")
        r2=clean_json(await call_openrouter(
            f"Write a {req.duration}s narration about '{req.topic or script[:80]}'. EXACTLY {tw_target} words. "
            f"Return JSON: {{\"script\":\"..words..\",\"scenes\":[{{\"description\":\"image prompt\",\"index\":0}},{{\"description\":\"image prompt\",\"index\":1}},{{\"description\":\"image prompt\",\"index\":2}},{{\"description\":\"image prompt\",\"index\":3}}]}}",
            "Return ONLY valid JSON."))
        try:
            d2=json.loads(r2); s2=d2.get("script","").strip()
            if len(s2.split())>len(script.split()): script=s2; raw_scenes=d2.get("scenes",raw_scenes)[:num_scenes]
        except: pass

    upd(16,f"✅ Script ready ({len(script.split())} words). Getting images...")
    image_pool=[]
    # User uploads
    if req.session_id:
        ud=OUTPUT_DIR/req.session_id/"uploads"
        if ud.exists():
            for ext in ["*.jpg","*.jpeg","*.png","*.webp"]: image_pool.extend(ud.glob(ext))

    # AI images (with Pexels video clips if PEXELS_KEY set)
    video_clips_pool = []
    PEXELS_KEY = os.environ.get("PEXELS_KEY", "")
    for i,scene in enumerate(raw_scenes):
        desc=scene.get("description",f"scene {i+1}")
        p=job_dir/f"ai_{i}.jpg"
        upd(16+int(i/len(raw_scenes)*22),f"🎨 AI image {i+1}/{len(raw_scenes)}...")
        if PEXELS_KEY:
            vp_clip=job_dir/f"pexels_{i}.mp4"
            kw=desc.split(",")[0][:50]
            if await fetch_pexels_video(kw, vp_clip): video_clips_pool.append(vp_clip)
        await download_image(f"{desc}, {img_style}",p,W,H)
        image_pool.append(p)
        if i<len(raw_scenes)-1: await asyncio.sleep(3)

    # Stock
    if req.use_stock:
        kw=NICHE_KW.get(req.niche,req.niche); fetched=0
        upd(38,f"📸 Fetching stock images...")
        for i in range(min(6,max(2,10-len(image_pool)))):
            sp=job_dir/f"stock_{i}.jpg"
            if await fetch_stock([kw,req.niche,"cinematic"][i%3],sp,W,H): image_pool.append(sp); fetched+=1
            await asyncio.sleep(1.5)

    if not image_pool: raise Exception("No images")
    user_imgs=[p for p in image_pool if "uploads" in str(p)]
    other_imgs=[p for p in image_pool if "uploads" not in str(p)]
    random.shuffle(other_imgs)
    image_pool=[]; ui=oi=slot=0
    while oi<len(other_imgs) or ui<len(user_imgs):
        if user_imgs and ui<len(user_imgs) and slot%3==0: image_pool.append(user_imgs[ui]); ui+=1
        elif oi<len(other_imgs): image_pool.append(other_imgs[oi]); oi+=1
        elif ui<len(user_imgs): image_pool.append(user_imgs[ui]); ui+=1
        slot+=1

    upd(42,"🎙️ Generating voiceover...")
    vp=job_dir/"voice.mp3"; await tts(script,req.voice,vp,req.voice_speed)
    dur=audio_dur(vp)
    upd(52,"🎵 Generating background music...")
    mp=job_dir/"music.wav"; has_mus=gen_music(dur+3,req.bg_music,mp)
    upd(58,"🎬 Building clips...")
    cuts=get_cut_times(dur,req.video_style)
    clips=[]
    for i,ts in enumerate(cuts):
        te=cuts[i+1] if i+1<len(cuts) else dur
        cp=job_dir/f"clip_{i:04d}.mp4"
        build_image_clip(image_pool[i%len(image_pool)],cp,W,H,max(0.1,te-ts),i,color_filter)
        clips.append(cp)
        if i%10==0: upd(58+int(i/len(cuts)*25),f"🎬 {i+1}/{len(cuts)} clips...")

    upd(83,"🔗 Stitching..."); cat=job_dir/"cat.mp4"
    if len(clips)==1: cat=clips[0]
    else: concat_clips(clips,cat)
    upd(90,"🎧 Final mix..."); srt=job_dir/"caps.srt"; create_srt(script,dur,srt)
    out=job_dir/"final.mp4"; merge_audio(cat,vp,mp if has_mus else None,srt,out,has_mus)
    return out

# ══════════════════════════════════════════════════════════════════
# TEMPLATE 2: IPHONE CHAT / INSTAGRAM DM  — complete rewrite
# Key fixes: duration control, real IG colors, short messages
# ══════════════════════════════════════════════════════════════════

def draw_ig_bubble(draw, img, text, y, W, is_sender, font):
    """Draw an Instagram-style bubble. Sender=purple gradient right, receiver=grey left."""
    lines = textwrap.wrap(text, width=42)
    line_h = 36
    pad_x, pad_y = 24, 14
    tw = max((draw.textlength(l, font=font) for l in lines), default=60)
    bw = min(int(tw) + pad_x * 2, W - 100)
    bh = len(lines) * line_h + pad_y * 2

    if is_sender:
        bx = W - bw - 24
        # Instagram purple gradient (simulate with solid #8B5CF6 → #EC4899)
        # Draw two-tone approximation
        for px in range(bw):
            ratio = px / bw
            r = int(139 + (236-139)*ratio)
            g = int(92  + (72 -92 )*ratio)
            b = int(246 + (153-246)*ratio)
            draw.rectangle([bx+px, y, bx+px+1, y+bh], fill=(r,g,b))
        # Rounded mask (draw white corners to simulate radius)
        draw.rounded_rectangle([bx, y, bx+bw, y+bh], radius=18, outline=(r,g,b), width=0)
        # Re-draw gradient properly via rounded_rectangle workaround
        # Use solid color that matches IG purple
        draw.rounded_rectangle([bx, y, bx+bw, y+bh], radius=18, fill=(168, 85, 247))
        tx, ty = bx + pad_x, y + pad_y
        for line in lines:
            draw.text((tx, ty), line, font=font, fill=(255,255,255))
            ty += line_h
    else:
        bx = 70
        draw.rounded_rectangle([bx, y, bx+bw, y+bh], radius=18, fill=(38, 38, 38))
        tx, ty = bx + pad_x, y + pad_y
        for line in lines:
            draw.text((tx, ty), line, font=font, fill=(255,255,255))
            ty += line_h

    return bh + 12


def render_ig_frame(messages_so_far, W, H, p1_name, p2_name, is_iphone=False):
    """Render a full Instagram DM or iMessage frame."""
    bg = (0, 0, 0) if not is_iphone else (18, 18, 18)
    img = Image.new("RGB", (W, H), bg)
    draw = ImageDraw.Draw(img)

    # ── Status bar ────────────────────────────────────────────────
    draw.rectangle([0, 0, W, 44], fill=(0,0,0))
    draw.text((40, 12), "9:41", font=get_font(26, True), fill=(255,255,255))
    draw.text((W-120, 12), "●●●", font=get_font(22), fill=(255,255,255))

    # ── Header ────────────────────────────────────────────────────
    hh = 110
    draw.rectangle([0, 44, W, hh], fill=(0,0,0))
    # Back chevron
    draw.text((20, 64), "‹", font=get_font(48, True), fill=(168,85,247))

    if is_iphone:
        # iMessage: contact centered, green/grey bubbles
        draw.text((W//2 - 60, 64), p2_name, font=get_font(32, True), fill=(255,255,255))
    else:
        # Instagram: username with @ and purple accent line at top
        draw.rectangle([0, 44, W, 47], fill=(168,85,247))
        # Small avatar circle
        av_x, av_y = W//2 - 26, 52
        draw.ellipse([av_x, av_y, av_x+52, av_y+52], fill=(80,40,120))
        draw.text((av_x+14, av_y+10), p2_name[0].upper() if p2_name else "A",
                  font=get_font(28, True), fill=(255,255,255))
        draw.text((W//2 - len(p2_name)*9, 108), p2_name,
                  font=get_font(28, True), fill=(255,255,255))

    # ── Thin separator ─────────────────────────────────────────────
    draw.rectangle([0, hh, W, hh+1], fill=(30,30,30))

    # ── Messages — render from bottom up so newest at bottom ──────
    mf = get_font(30)
    msg_area_h = H - hh - 20 - 90  # space for messages
    # Calculate total height needed
    rendered = []
    for msg in messages_so_far:
        lines = textwrap.wrap(msg["text"], width=42)
        bh = len(lines)*36 + 28 + 12
        rendered.append((msg, bh))

    # Scroll: only show messages that fit from bottom
    total_h = sum(h for _, h in rendered)
    skip_h = max(0, total_h - msg_area_h)
    y = hh + 20
    cum = 0
    for msg, bh in rendered:
        cum += bh
        if cum <= skip_h:
            continue
        is_sender = msg["sender"] == "p1"
        if is_iphone:
            # iMessage: blue sender, grey receiver
            lines = textwrap.wrap(msg["text"], width=42)
            pad_x, pad_y, line_h = 20, 12, 34
            tw = max((draw.textlength(l, font=mf) for l in lines), default=60)
            bw = min(int(tw) + pad_x*2, W-100)
            bub_h = len(lines)*line_h + pad_y*2
            bx = W-bw-20 if is_sender else 60
            color = (0,122,255) if is_sender else (38,38,38)
            draw.rounded_rectangle([bx,y,bx+bw,y+bub_h], radius=18, fill=color)
            ty = y+pad_y
            for line in lines:
                draw.text((bx+pad_x,ty),line,font=mf,fill=(255,255,255)); ty+=line_h
            y += bub_h+12
        else:
            y += draw_ig_bubble(draw, img, msg["text"], y, W, is_sender, mf)
        if y > H-100:
            break

    # ── Input bar ──────────────────────────────────────────────────
    bar_y = H - 80
    draw.rectangle([0, bar_y, W, H], fill=(0,0,0))
    draw.rectangle([0, bar_y, W, bar_y+1], fill=(30,30,30))
    if not is_iphone:
        # Camera icon left
        draw.text((24, bar_y+20), "📷", font=get_font(36), fill=(168,85,247))
        # Input box
        draw.rounded_rectangle([90, bar_y+14, W-24, H-14], radius=22, fill=(30,30,30))
        draw.text((110, bar_y+22), "Message...", font=get_font(28), fill=(80,80,80))
        # Heart emoji right
        draw.text((W-60, bar_y+20), "♡", font=get_font(36), fill=(168,85,247))
    else:
        draw.rounded_rectangle([60, bar_y+12, W-60, H-12], radius=22,
                                fill=(30,30,30), outline=(60,60,60), width=1)
        draw.text((80, bar_y+20), "iMessage", font=get_font(28), fill=(80,80,80))

    return img


async def render_chat_video(job_id, req, job_dir, upd):
    W, H = W9, H9
    is_iphone = req.template == "iphone_chat"
    is_ig = req.template == "instagram_dm"

    # ── Duration-aware message count ──────────────────────────────
    # Each message = ~2.5 words/sec at normal speed, max 12 words
    # 30s → ~8 msgs, 60s → ~14 msgs
    target_dur = req.duration
    # Words per message: short texts = 6-12 words
    words_per_msg = 8
    wps = 2.5 * req.voice_speed
    secs_per_msg = words_per_msg / wps   # ~3.2s per message
    num_msgs = max(6, min(20, int(target_dur / secs_per_msg)))
    max_words_per_msg = 12

    upd(5, f"✍️ Writing {num_msgs}-message conversation...")

    prompt = f"""Write a {req.tone} conversation between {req.person1_name} and {req.person2_name}.
Topic: {req.topic}

STRICT RULES — this becomes a video:
- Exactly {num_msgs} messages total, alternating p1/p2
- Each message: MAXIMUM {max_words_per_msg} words. Short texts only. Like real texting.
- No long paragraphs. Keep every message punchy and short.
- Make it emotional, dramatic, engaging — like viral TikTok chat videos
- Escalate tension through the conversation

Return ONLY a JSON array:
[{{"sender":"p1","text":"short message max {max_words_per_msg} words"}},{{"sender":"p2","text":"reply"}}]
Exactly {num_msgs} items. No markdown. No extra text."""

    raw = clean_json_array(await call_openrouter(
        prompt, "Return ONLY a JSON array starting with [ and ending with ]. No dict wrapper. No markdown."))
    messages = json.loads(raw)
    if isinstance(messages, dict):
        # Model wrapped array — find first list value
        for key in ("messages","conversation","chat","data","items","dialogue","msgs"):
            if key in messages and isinstance(messages[key], list):
                messages = messages[key]; break
        else:
            vals = [v for v in messages.values() if isinstance(v, list)]
            messages = vals[0] if vals else []
    if not isinstance(messages, list) or len(messages) == 0:
        raise Exception("Chat parse failed — not a list")

    # Enforce word limit per message
    messages = messages[:num_msgs]
    for m in messages:
        words = m.get("text","").split()
        if len(words) > max_words_per_msg:
            m["text"] = " ".join(words[:max_words_per_msg])

    upd(18, f"🎙️ Recording {len(messages)} voice lines...")

    audio_parts = []
    for i, msg in enumerate(messages):
        v = req.voice if msg["sender"] == "p1" else req.voice2
        p = job_dir / f"msg_{i:03d}.mp3"
        await tts(msg["text"], v, p, req.voice_speed)
        audio_parts.append((p, msg))
        if i % 4 == 0:
            upd(18 + int(i/len(messages)*20), f"🎙️ Voice {i+1}/{len(messages)}...")

    # Check total duration
    total_audio = sum(audio_dur(ap) for ap, _ in audio_parts)
    print(f"📊 Total audio: {total_audio:.1f}s (target: {target_dur}s, {len(messages)} msgs)")

    upd(38, "📱 Rendering Instagram DM frames...")
    frame_dir = job_dir/"frames"; frame_dir.mkdir(exist_ok=True)
    clip_paths = []; shown = []

    for i, (ap, msg) in enumerate(audio_parts):
        shown.append(msg)
        clip_dur = round(audio_dur(ap), 3)

        # Render frame
        frame_img = render_ig_frame(shown, W, H, req.person1_name, req.person2_name, is_iphone)
        fp = frame_dir / f"frame_{i:03d}.png"
        frame_img.save(str(fp))
        cp = job_dir / f"clip_{i:03d}.mp4"

        # ── Convert audio to WAV for maximum FFmpeg compatibility ────
        wav_p = job_dir / f"msg_{i:03d}.wav"
        rw = subprocess.run([
            "ffmpeg", "-y", "-i", str(ap),
            "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", str(wav_p)
        ], capture_output=True)
        audio_src = wav_p if (wav_p.exists() and wav_p.stat().st_size > 100) else ap

        clip_ok = False

        # Method 1: image + audio → libx264 + aac (clean, no duplicate -t)
        r1 = subprocess.run([
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(fp),
            "-i", str(audio_src),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "128k",
            "-pix_fmt", "yuv420p", "-r", "25",
            "-shortest", "-preset", "ultrafast", str(cp)
        ], capture_output=True)
        if cp.exists() and cp.stat().st_size > 1000:
            clip_ok = True
        else:
            print(f"⚠️ M1 err clip{i}: {r1.stderr.decode('utf-8',errors='replace')[-300:]}")

        if not clip_ok:
            # Method 2: two-step — silent video, then mux audio
            silent = job_dir / f"sil_{i:03d}.mp4"
            rs = subprocess.run([
                "ffmpeg", "-y", "-loop", "1", "-t", str(clip_dur), "-i", str(fp),
                "-c:v", "libx264", "-tune", "stillimage",
                "-pix_fmt", "yuv420p", "-r", "25", "-an",
                "-preset", "ultrafast", str(silent)
            ], capture_output=True)
            if silent.exists() and silent.stat().st_size > 100:
                rm = subprocess.run([
                    "ffmpeg", "-y", "-i", str(silent), "-i", str(audio_src),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                    "-map", "0:v", "-map", "1:a", "-shortest", str(cp)
                ], capture_output=True)
                if cp.exists() and cp.stat().st_size > 1000:
                    clip_ok = True
                else:
                    print(f"⚠️ M2b err: {rm.stderr.decode('utf-8',errors='replace')[-300:]}")
            else:
                print(f"⚠️ M2a err: {rs.stderr.decode('utf-8',errors='replace')[-300:]}")

        if not clip_ok:
            # Method 3: mpeg4 codec fallback (always in ffmpeg)
            r3 = subprocess.run([
                "ffmpeg", "-y",
                "-loop", "1", "-i", str(fp),
                "-i", str(audio_src),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "mpeg4", "-q:v", "5",
                "-c:a", "aac", "-b:a", "128k",
                "-pix_fmt", "yuv420p", "-r", "25",
                "-shortest", str(cp)
            ], capture_output=True)
            if cp.exists() and cp.stat().st_size > 1000:
                clip_ok = True
            else:
                print(f"⚠️ M3 err: {r3.stderr.decode('utf-8',errors='replace')[-300:]}")

        if clip_ok:
            clip_paths.append(cp)
        else:
            print(f"⚠️ Clip {i} failed all methods, skipping")

        upd(38 + int(i/len(messages)*38), f"📱 Frame {i+1}/{len(messages)}...")

    if not clip_paths:
        raise Exception("No chat clips created. Check FFmpeg codec support.")

    upd(76, "🔗 Stitching clips...")
    cat = job_dir/"cat.mp4"
    if len(clip_paths) == 1:
        cat = clip_paths[0]
    else:
        concat_clips(clip_paths, cat)

    upd(88, "🎵 Adding background music...")
    total_dur = sum(audio_dur(ap) for ap, _ in audio_parts)
    mp = job_dir/"music.wav"; has_mus = gen_music(total_dur + 3, req.bg_music, mp)
    out = job_dir/"final.mp4"

    if has_mus and mp.exists():
        r = subprocess.run([
            "ffmpeg", "-y", "-i", str(cat), "-i", str(mp),
            "-filter_complex", "[1:a]volume=0.15[m];[0:a][m]amix=inputs=2:duration=first[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-c:a", "aac", "-b:a", "128k", "-preset", "ultrafast", str(out)
        ], capture_output=True)
        if r.returncode != 0:
            subprocess.run(["ffmpeg","-y","-i",str(cat),"-c","copy",str(out)], capture_output=True)
    else:
        subprocess.run(["ffmpeg","-y","-i",str(cat),"-c","copy",str(out)], capture_output=True)

    return out


# ══════════════════════════════════════════════════════════════════
# TEMPLATE 3: REDDIT STORY
# ══════════════════════════════════════════════════════════════════
def render_reddit_card(title, body_preview, username, sub, votes, W, H):
    img=Image.new("RGB",(W,H),(18,18,18))
    draw=ImageDraw.Draw(img)
    # Reddit-style card
    cy=H//2-350; cw=W-60
    draw.rounded_rectangle([30,cy,30+cw,cy+700],radius=16,fill=(26,26,26))
    draw.rounded_rectangle([30,cy,30+cw,cy+702],radius=16,outline=(60,60,60),width=2)
    # Subreddit header
    draw.ellipse([52,cy+18,92,cy+58],fill=(255,69,0))
    draw.text((60,cy+26),"r",font=get_font(26,True),fill=(255,255,255))
    draw.text((100,cy+20),f"r/{sub}",font=get_font(28,True),fill=(255,255,255))
    draw.text((100,cy+52),f"Posted by u/{username}",font=get_font(24),fill=(130,130,130))
    # Awards
    draw.text((cw-80,cy+30),"🏆✨",font=get_font(28),fill=(255,215,0))
    # Title
    tf=get_font(34,True); ty=cy+100
    for line in textwrap.wrap(title,width=38):
        draw.text((52,ty),line,font=tf,fill=(215,215,215)); ty+=44
    # Body preview
    bf=get_font(28); ty+=10
    for line in textwrap.wrap(body_preview[:200],width=46):
        draw.text((52,ty),line,font=bf,fill=(160,160,160)); ty+=36
    # Vote bar
    vy=cy+640
    draw.rounded_rectangle([52,vy,52+80,vy+36],radius=18,fill=(255,69,0))
    draw.text((62,vy+4),f"↑ {votes}",font=get_font(24,True),fill=(255,255,255))
    draw.text((150,vy+8),"💬 Comments",font=get_font(24),fill=(130,130,130))
    draw.text((330,vy+8),"⬆ Share",font=get_font(24),fill=(130,130,130))
    return img

async def render_reddit_video(job_id, req, job_dir, upd):
    W,H=W9,H9
    upd(5,"📋 AI writing Reddit story...")
    wps=2.0*req.voice_speed; tw=int(req.duration*wps)
    prompt=f"""Write a compelling Reddit story for r/{req.subreddit}.
Topic: {req.topic} | Target words: {tw}
Write in first-person Reddit style. Include drama, emotion, conflict.
Return ONLY JSON:
{{"title":"AITA/reddit post title (dramatic hook)","story":"full {tw} word story as narration","body_preview":"first 40 words of story","votes":"{random.randint(12,89)}K"}}"""
    raw=clean_json(await call_openrouter(prompt,"Return ONLY valid JSON."))
    data=json.loads(raw)
    title=data.get("title","Reddit Story"); story=data.get("story","")
    body_preview=data.get("body_preview",story[:80]); votes=data.get("votes","42K")
    username=req.reddit_username or f"throwaway_{random.randint(1000,9999)}"

    upd(22,"🎨 Rendering Reddit card...")
    card=render_reddit_card(title,body_preview,username,req.subreddit,votes,W,H)
    card_path=job_dir/"reddit_card.png"; card.save(str(card_path))

    upd(35,"🎙️ Generating narration...")
    vp=job_dir/"voice.mp3"; await tts(story,req.voice,vp,req.voice_speed)
    dur=audio_dur(vp)

    upd(48,"🎵 Music..."); mp=job_dir/"music.wav"; has_mus=gen_music(dur+3,req.bg_music,mp)

    upd(55,"🎬 Building video...")
    # Start with reddit card for 4s, then background with scrolling story
    card_clip=job_dir/"card_clip.mp4"
    subprocess.run(["ffmpeg","-y","-loop","1","-i",str(card_path),"-t","4",
                    "-vf",f"scale={W}:{H},eq=saturation=1.1","-r","25",
                    "-c:v","libx264","-pix_fmt","yuv420p","-preset","ultrafast",str(card_clip)],
                   capture_output=True)

    # Rest of video: dark background with story text rolling (simulate teleprompter)
    words=story.split(); wps_actual=len(words)/max(dur-4,1)
    frame_dur=2.5; frames=[]; t=4.0
    word_idx=0
    while t<dur:
        chunk_words=words[word_idx:word_idx+18]; t+=frame_dur; word_idx=min(word_idx+18,len(words))
        text=" ".join(chunk_words)
        fimg=Image.new("RGB",(W,H),(18,18,18))
        fdraw=ImageDraw.Draw(fimg)
        # Subreddit watermark top
        fdraw.text((40,60),f"r/{req.subreddit}",font=get_font(32,True),fill=(255,69,0))
        fdraw.text((40,100),f"u/{username}",font=get_font(26),fill=(100,100,100))
        # Story text center
        ff=get_font(38); ty=H//2-200
        for line in textwrap.wrap(text,width=36):
            fdraw.text((50,ty),line,font=ff,fill=(220,220,220)); ty+=52
        fp=job_dir/f"story_frame_{len(frames):04d}.png"; fimg.save(str(fp)); frames.append((fp,frame_dur))

    story_clips=[]
    for i,(fp,fd) in enumerate(frames):
        cp=job_dir/f"story_clip_{i:04d}.mp4"
        subprocess.run(["ffmpeg","-y","-loop","1","-i",str(fp),"-t",str(fd),
                        "-vf",f"scale={W}:{H}","-r","25","-c:v","libx264",
                        "-pix_fmt","yuv420p","-preset","ultrafast",str(cp)],capture_output=True)
        story_clips.append(cp)

    all_clips=[card_clip]+story_clips; cat=job_dir/"cat.mp4"
    if len(all_clips)==1: cat=all_clips[0]
    else: concat_clips(all_clips,cat)

    upd(85,"🎧 Final mix..."); srt=job_dir/"caps.srt"; create_srt(story,dur,srt)
    out=job_dir/"final.mp4"; merge_audio(cat,vp,mp if has_mus else None,srt,out,has_mus)
    return out

# ══════════════════════════════════════════════════════════════════
# TEMPLATE 4: QUIZ / FACTS
# ══════════════════════════════════════════════════════════════════
def render_quiz_card(question, answer, reveal, W, H, niche):
    NICHE_COLORS={"tech":(0,150,255),"horror":(150,0,0),"motivation":(255,140,0),
                  "space":(100,0,200),"finance":(0,180,100),"default":(255,69,0)}
    accent=NICHE_COLORS.get(niche,NICHE_COLORS["default"])
    img=Image.new("RGB",(W,H),(12,12,15))
    draw=ImageDraw.Draw(img)
    # Background gradient effect (simple top strip)
    for i in range(300):
        alpha=1-i/300; c=tuple(int(x*alpha) for x in accent)
        draw.line([(0,i),(W,i)],fill=c)
    # "?" or answer icon
    if not reveal:
        draw.ellipse([W//2-100,200,W//2+100,400],fill=accent)
        draw.text((W//2-40,250),"?",font=get_font(120,True),fill=(255,255,255))
    else:
        draw.ellipse([W//2-100,200,W//2+100,400],fill=(0,200,100))
        draw.text((W//2-50,250),"✓",font=get_font(100,True),fill=(255,255,255))
    # Question
    qf=get_font(44,True); qy=440
    for line in textwrap.wrap(question,width=28):
        qw=draw.textlength(line,font=qf)
        draw.text(((W-qw)//2,qy),line,font=qf,fill=(255,255,255)); qy+=58
    if reveal:
        # Answer box
        draw.rounded_rectangle([60,qy+40,W-60,qy+200],radius=20,fill=(0,40,0))
        draw.rounded_rectangle([60,qy+40,W-60,qy+202],radius=20,outline=(0,200,100),width=3)
        draw.text((80,qy+60),"ANSWER:",font=get_font(28,True),fill=(0,200,100))
        af=get_font(40,True); ay=qy+100
        for line in textwrap.wrap(answer,width=30):
            draw.text((80,ay),line,font=af,fill=(255,255,255)); ay+=50
    return img

async def render_quiz_video(job_id, req, job_dir, upd):
    W,H=W9,H9
    upd(5,"🧠 AI generating quiz questions...")
    nq=req.num_questions
    prompt=f"""Generate {nq} fascinating quiz questions about: {req.topic}
Niche: {req.niche}. Make answers surprising/shocking for viral appeal.
Return ONLY JSON array: [{{"question":"...?","answer":"...","fun_fact":"one additional interesting sentence"}}]"""
    raw=clean_json(await call_openrouter(prompt,"Return ONLY valid JSON array."))
    qas=json.loads(raw)
    if not isinstance(qas,list): raise Exception("Quiz parse failed")
    qas=qas[:nq]

    upd(20,"🎙️ Recording voiceovers...")
    clips=[]; total_dur=0
    for i,qa in enumerate(qas):
        q=qa.get("question",""); a=qa.get("answer",""); ff=qa.get("fun_fact","")
        # Question TTS
        qp=job_dir/f"q_{i}.mp3"; await tts(f"Question {i+1}. {q}",req.voice,qp,req.voice_speed)
        qd=audio_dur(qp)
        # "Think about it..." pause = 2s silence
        # Answer TTS
        ap=job_dir/f"a_{i}.mp3"; await tts(f"The answer is... {a}. {ff}",req.voice,ap,req.voice_speed)
        ad=audio_dur(ap)
        # Render question frame
        qf=render_quiz_card(q,"",False,W,H,req.niche)
        qfp=job_dir/f"qf_{i}.png"; qf.save(str(qfp))
        # Render answer frame
        af=render_quiz_card(q,a,True,W,H,req.niche)
        afp=job_dir/f"af_{i}.png"; af.save(str(afp))
        # Build clips
        qcp=job_dir/f"qclip_{i:03d}.mp4"
        subprocess.run(["ffmpeg","-y","-loop","1","-i",str(qfp),"-i",str(qp),
                        "-c:v","libx264","-c:a","aac","-pix_fmt","yuv420p",
                        "-shortest","-preset","ultrafast",str(qcp)],capture_output=True)
        # 1.5s suspense pause clip
        scp=job_dir/f"sclip_{i:03d}.mp4"
        subprocess.run(["ffmpeg","-y","-loop","1","-i",str(qfp),"-t","1.5",
                        "-vf",f"scale={W}:{H}","-r","25","-c:v","libx264",
                        "-pix_fmt","yuv420p","-an","-preset","ultrafast",str(scp)],capture_output=True)
        # Answer clip
        acp=job_dir/f"aclip_{i:03d}.mp4"
        subprocess.run(["ffmpeg","-y","-loop","1","-i",str(afp),"-i",str(ap),
                        "-c:v","libx264","-c:a","aac","-pix_fmt","yuv420p",
                        "-shortest","-preset","ultrafast",str(acp)],capture_output=True)
        clips+=[qcp,scp,acp]; total_dur+=qd+1.5+ad
        upd(20+int(i/len(qas)*55),f"🧠 Q{i+1}/{len(qas)} done...")

    upd(75,"🔗 Stitching quiz..."); cat=job_dir/"cat.mp4"
    if len(clips)==1: cat=clips[0]
    else: concat_clips(clips,cat)
    upd(85,"🎵 Music...")
    mp=job_dir/"music.wav"; has_mus=gen_music(total_dur+3,req.bg_music,mp)
    out=job_dir/"final.mp4"
    if has_mus:
        r=subprocess.run(["ffmpeg","-y","-i",str(cat),"-i",str(mp),
            "-filter_complex","[1:a]volume=0.15[m];[0:a][m]amix=inputs=2:duration=first[aout]",
            "-map","0:v","-map","[aout]","-c:v","libx264","-c:a","aac","-b:a","128k",
            "-preset","ultrafast",str(out)],capture_output=True)
        if r.returncode!=0: subprocess.run(["ffmpeg","-y","-i",str(cat),"-c","copy",str(out)],capture_output=True)
    else:
        subprocess.run(["ffmpeg","-y","-i",str(cat),"-c","copy",str(out)],capture_output=True)
    return out

# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE DISPATCHER
# ══════════════════════════════════════════════════════════════════
async def run_pipeline(job_id: str, req: VideoRequest):
    job_dir=OUTPUT_DIR/job_id; job_dir.mkdir(parents=True,exist_ok=True)
    def upd(pct, msg):
        jobs[job_id].update({"progress":pct,"message":msg})
        print(f"[{job_id[:8]}] {pct}% {msg}")
    try:
        jobs[job_id]["status"]="processing"
        t=req.template
        if t in ("iphone_chat","instagram_dm"): out=await render_chat_video(job_id,req,job_dir,upd)
        elif t=="reddit_story": out=await render_reddit_video(job_id,req,job_dir,upd)
        elif t=="quiz": out=await render_quiz_video(job_id,req,job_dir,upd)
        else: out=await render_cinematic(job_id,req,job_dir,upd)
        jobs[job_id].update({"status":"done","video_path":str(out)})
        upd(100,"✅ Video ready!")
    except Exception as e:
        jobs[job_id].update({"status":"error","message":f"❌ {e}"})
        print(f"[{job_id[:8]}] ERROR: {e}")

# ─── ROUTES ───────────────────────────────────────────────────────
@app.get("/") 
async def home(): return FileResponse("static/index.html")
@app.get("/dashboard") 
async def dashboard(): return FileResponse("static/dashboard.html")

@app.post("/api/upload-images/{session_id}")
async def upload_images(session_id: str, files: List[UploadFile]=File(...)):
    d=OUTPUT_DIR/session_id/"uploads"; d.mkdir(parents=True,exist_ok=True)
    saved=[]
    for f in files:
        if f.content_type not in ["image/jpeg","image/png","image/webp","image/jpg"]: continue
        p=d/f"{uuid.uuid4().hex[:8]}_{f.filename}"; p.write_bytes(await f.read()); saved.append(f.filename)
    return {"uploaded":len(saved),"session_id":session_id}

@app.post("/api/generate")
async def generate(req: VideoRequest, bg: BackgroundTasks):
    jid=str(uuid.uuid4())
    jobs[jid]={"status":"queued","progress":0,"message":"⏳ Starting...","video_path":None}
    bg.add_task(run_pipeline,jid,req)
    return {"job_id":jid}

@app.get("/api/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs: return JSONResponse({"error":"Not found"},status_code=404)
    return jobs[job_id]

@app.get("/api/download/{job_id}")
async def download(job_id: str):
    j=jobs.get(job_id)
    if not j or j["status"]!="done": return JSONResponse({"error":"Not ready"},status_code=400)
    return FileResponse(j["video_path"],media_type="video/mp4",filename="shortsai_video.mp4")


@app.get("/api/services")
async def check_services():
    """Returns which free services are configured/available."""
    has_pexels = bool(os.environ.get("PEXELS_KEY", ""))
    has_hf = bool(os.environ.get("HF_TOKEN", ""))
    has_or = any(os.environ.get(f"OR_KEY_{i}","").strip() for i in range(1,4))
    return {
        "openrouter": has_or,
        "pollinations": True,   # always tried first
        "hf_images": has_hf,    # better with token, works without for some models
        "pexels_video": has_pexels,
        "tts": True,            # edge-tts, always free
        "notes": {
            "pexels_video": "Add PEXELS_KEY env var (free at pexels.com/api) for real video clips",
            "hf_images": "Add HF_TOKEN env var (free at huggingface.co) for better image fallback",
        }
    }

app.mount("/static",StaticFiles(directory="static"),name="static")
