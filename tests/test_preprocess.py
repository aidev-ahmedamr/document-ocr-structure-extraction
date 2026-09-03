from PIL import Image, ImageDraw

from src.preprocessing.preprocess import preprocess_document


def test_preprocess_single_image(tmp_path):
    img_path = tmp_path / "sample.png"
    img = Image.new("RGB", (300, 150), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 60), "TEST DOCUMENT", fill="black")
    img.save(img_path)

    pages = preprocess_document(str(img_path))

    assert len(pages) == 1
    assert pages[0].mode == "L"  # binarized grayscale
