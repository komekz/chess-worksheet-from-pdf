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
parser.add_argument("--start-page", "-s", default=0, type=int, help="First page that contains puzzles (used for correct page numbering)")

args = parser.parse_args()

def main():

    PDF_PATH = Path(args.input_pdf)
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f'This file does not exist: {PDF_PATH}')
    if PDF_PATH.suffix != '.pdf':
        raise ValueError(f"{RED}excepts a .pdf file, received: {PDF_PATH.suffix}{RESET}")
    PUZZLE_NAME = args.name
    START_PAGE_NUM = args.start_page

    RESULT_DIR = Path('result')

    with ChessPDF(str(PDF_PATH), PUZZLE_NAME, START_PAGE_NUM) as chessPDF:

        for png, page_num in tqdm(chessPDF, desc="Processing PDF Pages"):
            recContours = getRectangleContours(png)
            cutPuzzles = cropPuzzles(recContours, png)

            FENs = []
            for puzzle in cutPuzzles:
                result = get_fen(puzzle, "chess")

                if result:
                    fen = result.fen
                    FENs.append(fen)

                    # chessBoardVis(fen, puzzle)

            # There must be exactly 6 puzzles to print a worksheet
            if len(FENs) != 6:
                continue

            pdf_file_name = RESULT_DIR / f'result/{PUZZLE_NAME}-{page_num}.pdf'
            create_worksheet_pdf(
                json.dumps(FENs), 
                page_num, 
                str(pdf_file_name))

        # plt.show()

def cropPuzzles(recContours, pillowImg):
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
        clearance = 15
        left = max(0, np.min(contour[:,:, 0]) - clearance)
        right = min(max_w, np.max(contour[:,:, 0]) + clearance)
        top = max(0, np.min(contour[:,:, 1]) - clearance)
        bottom = min(max_h, np.max(contour[:,:, 1]) + clearance)

        cropPuzzle = pillowImg.crop((left, top, right, bottom))
        cutPuzzles.append(cropPuzzle)

        # for manually verifying if the fen are correct
        # cropPuzzle.save(f'test_imgs/puzzle-{len(cutPuzzles)}.png')

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
