import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import chess
import chess.svg
from cairosvg import svg2png
import json
from tqdm import tqdm
from pathlib import Path
import argparse
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

    with ChessPDF(str(PDF_PATH), PUZZLE_NAME, EMPTY_PAGES) as chessPDF:
        for png, page_num in tqdm(chessPDF, desc="Processing PDF Pages"):
            recContours = getRectangleContours(png)
            cutPuzzles = cropPuzzles(recContours, png)

            FENs = []
            actualPuzzles = []
            for puzzle in cutPuzzles:
                result = get_fen(puzzle["img"], "chess")

                if result and result.fen:
                    fen = result.fen
                    FENs.append(fen)
                    actualPuzzles.append(puzzle)

                    # chessBoardVis(fen, puzzle)

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

            # pdf_file_name = RESULT_DIR / f'{PUZZLE_NAME}-{page_num}.pdf'
            # create_worksheet_pdf(
            #     json.dumps(FENs), 
            #     page_num, 
            #     str(pdf_file_name))
    
    page_num = 1

    while page_num * 6 <= len(allFEN):
        pdf_file_name = RESULT_DIR / f'{PUZZLE_NAME}-{page_num}.pdf'

        start = (page_num - 1) * 6
        end = page_num * 6
        create_worksheet_pdf(
            json.dumps(allFEN[start:end]), 
            page_num, 
            str(pdf_file_name))
        page_num += 1

        # plt.show()

def findTemplate(gray_image, template):
    best_confidence = 0
    best_location = None
    best_scale = None

    for scale in np.arange(0.5, 1.1, 0.05):
        resized = cv2.resize(
            template,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

        # Template must be smaller than source image
        if (
            resized.shape[0] > gray_image.shape[0]
            or resized.shape[1] > gray_image.shape[1]
        ):
            continue

        result = cv2.matchTemplate(
            gray_image,
            resized,
            cv2.TM_CCOEFF_NORMED
        )

        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val > best_confidence:
            best_confidence = max_val
            best_location = max_loc
            best_scale = scale

    return best_confidence, best_location, best_scale

def setSideToPlay(puzzles: list[dict], FENs: list[str]) -> list:
    for index, puzzle in enumerate(puzzles):
        puzzle["img"].save(f"{index}.png")
        # Convert to numpy array for cv2
        img = np.array(puzzle["img"])
        # 1. Load the main puzzle image and convert it to grayscale
        # (Ideally, pass just the cropped bottom-right corner of the board to make this lightning fast)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2. Load your cropped black triangle template in grayscale
        triangle_template = cv2.imread('templates/black_triangle_template.png', 0)
        text_template = cv2.imread('templates/black_to_move_text.png', 0)
        assert triangle_template is not None
        assert text_template is not None

        template = triangle_template if args.template == "triangle" else text_template

        threshold = 0.80

        confidence, _, _ = findTemplate(gray_image, template)

        print(f'highest match confidence: {confidence}')

        # 6. Evaluate the result
        if confidence >= threshold:
            fenList = FENs[index].split(" ")
            fenList[1] = 'b'
            FENs[index] = ' '.join(fenList)

    return FENs
            

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

        # Give each crop edge some offset, capped at the image borders
        clearance = 20
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

def getRectangleContours(pillowImg):
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

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Only allow contours with 4 edges (rectangle)
    recContours = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            recContours.append(approx)

    return recContours

# Show the original board crop alongside the inferred board position
def chessBoardVis(fen: str, originalBd):
    if len(fen) <= 5: 
        return
    board = chess.Board(fen)
    svg_string = chess.svg.board(board, size=400)
    svg2png(bytestring=svg_string, write_to='chessboard.png')

    inferredBd = Image.open('chessboard.png')

    plt.figure(figsize=(6,3))
    plt.subplot(121)
    plt.imshow(originalBd)
    plt.subplot(122)
    plt.imshow(inferredBd)
    plt.title(f'{fen.split(" ")[1]} to move')


if __name__ == "__main__":
    main()
