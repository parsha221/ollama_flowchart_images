import fitz  # PyMuPDF
import os

def extract_images(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    try:
        pdf_document = fitz.open(pdf_path)
        print(f"Opened PDF: {pdf_path}")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            images = page.get_images(full=True)
            print(f"Page {page_num+1}: Found {len(images)} images")
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                img_filename = f"image_{page_num+1}_{img_index+1}.{image_ext}"
                with open(img_filename, "wb") as img_file:
                    img_file.write(image_bytes)
                print(f"Extracted {img_filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

extract_images(r"C:\Users\2216184\OneDrive - Cognizant\Desktop\i_test.pdf")
