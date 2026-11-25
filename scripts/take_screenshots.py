"""Capture screenshots of the running Streamlit app for the report/presentation (dev tool).

Requires: pip install playwright   (uses the locally installed Google Chrome, no browser download)
Usage:    python scripts/take_screenshots.py
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
PORT = 8599


def wait_for_server(timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if b"ok" in urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2).read():
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError("Streamlit did not start")


def main():
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    samples = sorted(p.name for p in (ROOT / "sample_images").glob("*.jpg"))
    picks = [s for s in samples if s.startswith("mel")][:1] + [s for s in samples if s.startswith("nv")][:1]
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
                             "--server.headless", "true", "--server.port", str(PORT),
                             "--browser.gatherUsageStats", "false", "--theme.base", "light"],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_server()
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome", headless=True)
            page = browser.new_page(viewport={"width": 1500, "height": 1250}, device_scale_factor=1)
            for i, sample in enumerate(picks, 1):
                page.goto(f"http://localhost:{PORT}/?sample={sample}")
                page.wait_for_selector("text=Confidence scores", timeout=120000)
                page.wait_for_selector("[data-testid='stImage'] img", timeout=60000)
                time.sleep(4)
                out = OUT / f"app_demo_{i}_{sample.split('_')[0]}.png"
                page.screenshot(path=str(out), full_page=True)
                print("saved", out)
            browser.close()
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
