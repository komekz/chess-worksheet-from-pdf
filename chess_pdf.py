import pymupdf
from tqdm import tqdm
from PIL import Image


class ChessPDF:
    def __init__(self, 
                 pdf_path: str, 
                 puzzle_name: str, 
                 start_page: int = 0
                ) -> None:
        self.doc = pymupdf.open(pdf_path)
        self.current_page_num = start_page
        self.total_pages = self.doc.page_count - start_page
        self.start_page_num = start_page
        self.puzzle_name = puzzle_name

    def __len__(self):
        return self.total_pages

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_page_num == self.total_pages:
            raise StopIteration
        
        current_page = self.doc[self.current_page_num]
        self.current_page_num += 1

        pix = current_page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0))

        mode = "RGBA" if pix.alpha else "RGB"
        # Convert the PyMuPDF Pixmap to a Pillow Image
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)

        return (img, (self.current_page_num - self.start_page_num))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.doc.close()






