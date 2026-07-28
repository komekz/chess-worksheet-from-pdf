import os
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import chess
import chess.svg
from cairosvg import svg2png
import json
from tqdm import tqdm

from fetch_pdf import create_worksheet_pdf
from chess_pdf import ChessPDF

# For the chess lib to import properly
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(ROOT_DIR, "chess_fen_lib")
sys.path.insert(0, LIB_DIR)

from chess_fen_lib.chess_diagram_to_fen import get_fen
from PIL import Image

def main():

    PDF_PATH = '/Users/james/Downloads/book1.pdf'
    PUZZLE_NAME = 'book-1'


    with ChessPDF(PDF_PATH, PUZZLE_NAME, 4) as chessPDF:

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

            create_worksheet_pdf(
                json.dumps(FENs), 
                page_num, 
                f'test_imgs/{PUZZLE_NAME}-{page_num}.pdf')

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
