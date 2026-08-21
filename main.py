import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
from tqdm import tqdm
from pathlib import Path
import argparse
import easyocr
from fetch_pdf import create_worksheet_pdf
from chess_pdf import ChessPDF

# For the chess lib to import properly
ROOT_DIR = Path(__file__).parent.resolve()
LIB_DIR = ROOT_DIR / "chess_fen_lib"
sys.path.insert(0, str(LIB_DIR))

from chess_fen_lib.chess_diagram_to_fen import get_fen
from PIL import Image

RED = "\033[31m"
RESET = "\033[0m"

parser = argparse.ArgumentParser(description='''
    Turns multi page unstructured pdfs into individual worksheets
''')
parser.add_argument("input_pdf", help="the PDF to turn into worksheets")
parser.add_argument("--name", "-n", required=True, help="name of the output worksheets")
parser.add_argument("--empty-pages", "-e", default=0, type=int, help="number of un-numbered pages at the start of the pdf (used for correct page numbering)")
parser.add_argument("--puzzles-per-page", "-p", default=-1, type=int, help="number of puzzles per page if it is the same across all pages. leave blank for variable puzzles per page")
parser.add_argument("--template", "-t", required=True, choices=["triangle", "text"], help="select the indicator for side to play.")

args = parser.parse_args()

def main():

    PDF_PATH = Path(args.input_pdf)
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f'{RED}This file does not exist: {PDF_PATH}{RESET}')
    if PDF_PATH.suffix != '.pdf':
        raise ValueError(f"{RED}excepts a .pdf file, received: {PDF_PATH.suffix}{RESET}")
    PUZZLE_NAME = args.name
    EMPTY_PAGES = args.empty_pages
    PUZZLES_PER_PAGE = args.puzzles_per_page

    RESULT_DIR = Path(f'result/{PUZZLE_NAME}')
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    allFEN = []

    puzzle_page_num = 1
    with ChessPDF(str(PDF_PATH), PUZZLE_NAME, EMPTY_PAGES) as chessPDF:
        for png, page_num in tqdm(chessPDF, desc="Processing PDF Pages"):
            recContours = getSquareContours(png)
            cutPuzzles = cropPuzzles(recContours, png)

            FENs = []
            actualPuzzles = []
            for puzzle in cutPuzzles:
                result = get_fen(puzzle["img"], "chess")

                if result and result.fen:
                    fen = result.fen
                    FENs.append(fen)
                    actualPuzzles.append(puzzle)

            # enforce specific number of puzzles per page, if applicable
            if (PUZZLES_PER_PAGE > 0) and (len(FENs) != PUZZLES_PER_PAGE):
                continue

            # Sort by x position
            # i = 0
            # while i < len(actualPuzzles):
            #     curX = actualPuzzles[i]["x"]
            #     nextX = actualPuzzles[i+1]["x"]
            #     if (curX > nextX):
            #         FENs[i], FENs[i+1] = FENs[i+1], FENs[i]
            #         actualPuzzles[i], actualPuzzles[i+1] = actualPuzzles[i+1], actualPuzzles[i]
            #     i += 2

            # Set side to play based on indicator
            FENs = setSideToPlay(actualPuzzles, FENs)
            allFEN.extend(FENs)
    
            # Create as many pages as it can
            while len(allFEN) >= 6:
                pdf_file_name = RESULT_DIR / f'{PUZZLE_NAME}-{puzzle_page_num}.pdf'

                create_worksheet_pdf(json.dumps(allFEN[0:6]), page_num, str(pdf_file_name))

                del allFEN[0:6]
                puzzle_page_num += 1

def blackToPlayFEN(fen: str) -> str:
    fenList = fen.split(" ")
    fenList[1] = 'b'

    return ' '.join(fenList)

def setSideToPlay(puzzles: list[dict], FENs: list[str]) -> list:
    for index, puzzle in enumerate(puzzles):
        # Convert to numpy array for cv2
        img = np.array(puzzle["img"])
        # 1. Load the main puzzle image and convert it to grayscale
        # (Ideally, pass just the cropped bottom-right corner of the board to make this lightning fast)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if args.template == 'text':
            if containsBlackText(gray_image):
                FENs[index] = blackToPlayFEN(FENs[index])
        # elif args.template == 'triangle':
            
    return FENs

reader = easyocr.Reader(['ch_sim','en'])
BLACK_STRINGS = ['黑', 'black']
def containsBlackText(img) -> bool:
    results = reader.readtext(img, detail=0)

    for string in results:
        if any(sub in string for sub in BLACK_STRINGS):
            return True

    return False
            
def cropPuzzles(recContours, pillowImg) -> list:
    '''
    for each rectangle, crop an area from the original image 
    that is slightly larger to preserve board edges. This
    is an image containing a single puzzle.

    For the rectangles that aren't actually chessboards, 
    get_fen should just return None
    '''

    cutPuzzles = []
    for contour in recContours:
        max_w, max_h = pillowImg.size

        # Give each crop edge some offset so the model has some room. capped at the image borders
        clearance = 25
        left = max(0, np.min(contour[:,:, 0]) - clearance)
        right = min(max_w, np.max(contour[:,:, 0]) + clearance)
        top = max(0, np.min(contour[:,:, 1]) - clearance)
        bottom = min(max_h, np.max(contour[:,:, 1]) + clearance + 25)

        cropPuzzle = pillowImg.crop((left, top, right, bottom))
        cutPuzzles.append({
            "img": cropPuzzle, 
            "x": left, 
            "y": top
        })

    # sorted by y position, puzzles at the top of the page starts first
    cutPuzzles.sort(key=(lambda x : x["y"]))

    return cutPuzzles

i = 0
def getSquareContours(pillowImg):
    '''
    find all the squares in the uploaded image, then
    crop each of them with an offset and save to an array
    '''

    # Convert pillow image to numpy array for cv2
    img = np.array(pillowImg)

    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Canny only takes grayscale, so convert first
    imgGray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
    # detect all the edges and turn image into binary for findContours
    edges = cv2.Canny(imgGray, 50, 150)

    kernel = np.ones((3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # filter contours in terms of their size and how square-like they are
    recContours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20000: continue

        x, y, w, h= cv2.boundingRect(contour)
        ratio = w / h

        if 0.8 < ratio < 1.2:
            recContours.append(contour)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)

            # DEBUG
            cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    # DEBUG
    global i
    cv2.imwrite(f"contourstest/{i}.png", img)
    i += 1

    return recContours


if __name__ == "__main__":
    main()
