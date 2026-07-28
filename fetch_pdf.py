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

        # 3. Load the page
        page.goto(react_url)
        
        # 4. Wait for React to finish rendering
        page.wait_for_selector(".chessboards")
        
        # 5. Trigger the headless "window.print()" and save directly to disk
        page.pdf(
            path=output_filepath
        )
        
        # 6. Cleanup
        browser.close()
