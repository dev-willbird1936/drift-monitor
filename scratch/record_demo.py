"""Record the demo page in a real browser and save an mp4.

usage: record_demo.py URL OUT_MP4 [replay|live] [chip title]
replay: presses Play on a replay and records until it finishes.
live:   runs the live tab once (needs ?scorer=... in URL and a key server on 127.0.0.1:8801, see publish/keyserve.py).
"""
import os, shutil, subprocess, sys, tempfile, time
from playwright.sync_api import sync_playwright

url, out = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "replay"
chip = sys.argv[4] if len(sys.argv) > 4 else "Extra files"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

tmp = tempfile.mkdtemp(prefix="drift-rec-")
with sync_playwright() as p:
    exe = os.path.expanduser("~/AppData/Local/ms-playwright/chromium_headless_shell-1254/chrome-headless-shell-win64/chrome-headless-shell.exe")
    b = p.chromium.launch(executable_path=exe if os.path.exists(exe) else None)
    ctx = b.new_context(viewport={"width": 1280, "height": 800}, record_video_dir=tmp, record_video_size={"width": 1280, "height": 800})
    page = ctx.new_page()
    page.goto(url)
    page.wait_for_timeout(1500)
    if mode == "replay":
        page.get_by_role("button", name=chip, exact=True).click()
        page.wait_for_timeout(1500)
        page.click("#bigplay")
        t0 = time.time()
        while time.time() - t0 < 240:
            page.wait_for_timeout(500)
            if page.locator("#end").is_visible():
                break
        page.wait_for_timeout(1500)
        page.mouse.wheel(0, 700)
        page.wait_for_timeout(3000)
    else:
        page.click("button.tab[data-v=live]")
        page.wait_for_timeout(1500)
        page.select_option("#push", "extra")
        page.evaluate("async () => { document.getElementById('key').value = await (await fetch('http://127.0.0.1:8801/k')).text(); }")
        page.click("#run")
        t0 = time.time()
        while time.time() - t0 < 300:
            page.wait_for_timeout(500)
            if page.locator("#run").is_visible():
                break
        page.evaluate("() => { document.getElementById('key').value = ''; }")
        page.wait_for_timeout(2000)
        page.mouse.wheel(0, 700)
        page.wait_for_timeout(3000)
    video = page.video.path()
    ctx.close()
    b.close()
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", video, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart", out], check=True)
print("wrote", out, os.path.getsize(out) // 1024, "KB")
