"""Test launching Streamlit and capturing real screenshots using Edge."""

import os
import subprocess
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

EDGE_PATH = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
SCREENSHOT_INITIAL = Path("reports") / "streamlit_ui_initial.png"
SCREENSHOT_ANALYZED = Path("reports") / "streamlit_ui_analyzed.png"

def log(msg: str) -> None:
    print(msg, flush=True)

def main() -> None:
    log("=== Starting Streamlit UI Capture Test ===")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    server_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port=8501",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    log(f"Launching server: {' '.join(server_cmd)}")
    proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path.cwd()),
        env=env,
    )
    
    # Wait 6 seconds for Streamlit server initialization
    time.sleep(6)
    
    try:
        log("Launching browser with --no-sandbox...")
        with sync_playwright() as p:
            browser = p.chromium.launch(
                executable_path=EDGE_PATH,
                headless=True,
                args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
            )
            context = browser.new_context(viewport={"width": 1280, "height": 950})
            page = context.new_page()
            
            log("Navigating to http://localhost:8501 ...")
            page.goto("http://localhost:8501", timeout=30000)
            page.wait_for_timeout(4000)
            
            # Check Title
            h1 = page.locator("h1").first.inner_text()
            log(f"Verified Page Header: '{h1}'")
            
            SCREENSHOT_INITIAL.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(SCREENSHOT_INITIAL), full_page=False)
            log(f"Captured initial state screenshot: {SCREENSHOT_INITIAL}")
            
            # Verify no prediction is visible initially
            pred_count_initial = page.locator("text=Siren Detected").count() + page.locator("text=No Siren Detected").count()
            log(f"Prediction cards count before audio selection: {pred_count_initial}")
            assert pred_count_initial == 0, "No prediction should be visible on initial load!"
            
            # Open 'Try sample audio' expander
            log("Opening 'Try sample audio' expander...")
            expander = page.locator("text=Try sample audio").first
            expander.click()
            page.wait_for_timeout(1000)
            
            # Select Emergency Siren sample
            log("Selecting benchmark Emergency Siren sample...")
            page.locator("label:has-text('Emergency Siren')").first.click()
            page.wait_for_timeout(2000)
            
            # Verify STILL NO prediction is displayed before clicking Analyze!
            pred_count_before_analyze = page.locator("text=Siren Detected").count() + page.locator("text=No Siren Detected").count()
            log(f"Prediction cards count after selecting audio but BEFORE clicking Analyze: {pred_count_before_analyze}")
            assert pred_count_before_analyze == 0, "CRITICAL: Prediction must NOT display before clicking Analyze Audio!"
            
            # Click 'Analyze Audio' button
            log("Clicking 'Analyze Audio' button...")
            page.locator("button:has-text('Analyze Audio')").first.click()
            
            # Wait for result to appear
            log("Waiting for inference completion and dynamic charts...")
            page.wait_for_selector("text=Siren Detected", timeout=25000)
            page.wait_for_timeout(2500)
            
            # Capture analyzed screenshot
            page.screenshot(path=str(SCREENSHOT_ANALYZED), full_page=False)
            log(f"Captured analyzed state screenshot: {SCREENSHOT_ANALYZED}")
            
            # Verify probability text exists
            prob_text = page.locator("text=Model Estimated Siren Probability").first.inner_text()
            log(f"Verified Detection Result: '{prob_text}'")
            
            browser.close()
            log("=== UI VERIFICATION TEST PASSED SUCCESSFULLY ===")
    finally:
        log("Terminating server process...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

if __name__ == "__main__":
    main()
