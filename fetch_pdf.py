import urllib.parse
from playwright.sync_api import sync_playwright
from tqdm import tqdm

def create_worksheet_pdf(fen_json: str, page_num: int, output_filepath: str):
    with sync_playwright() as p:
        # 1. Spin up an invisible browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # 2. Inject the FEN by appending it to the React app's local URL
        # We url-encode it so spaces and slashes in the FEN don't break the web address
        encoded_params = urllib.parse.urlencode({
            "fen": fen_json, 
            "pageNum": page_num
        })
        react_url = f"https://chessworksheets.me/?{encoded_params}"

        try:
            page.goto(react_url)
            
            # 4. Wait for React to finish rendering
            page.wait_for_selector(".chessboards", timeout=30000)
            
            # 5. Trigger the headless "window.print()" and save directly to disk
            page.pdf(path=output_filepath)
            
        except PlaywrightTimeoutError:
            print(f"\n[ERROR] Playwright timed out on page {page_num}.")
            print(f"Failed URL: {react_url}")
            print("Skipping this page and continuing...\n")
            
        finally:
            # 6. Cleanup (Ensure the browser always closes, even on failure)
            browser.close()
