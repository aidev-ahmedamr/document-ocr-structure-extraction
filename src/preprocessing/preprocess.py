import numpy as np
from PIL import Image, ImageOps
from pdf2image import convert_from_path


def load_document_as_images(file_path, dpi=200):
    """
    Turn any supported input (PDF or image) into a list of PIL images,
    one per page. Single images become a one-item list.
    """

    if file_path.lower().endswith(".pdf"):
        return convert_from_path(file_path, dpi=dpi)

    return [Image.open(file_path).convert("RGB")]


def deskew(image):
    """
    Estimate and correct small rotation using the orientation of the
    document's own dark (text) pixels - no external skew-detection
    library needed.
    """

    grayscale = image.convert("L")
    array = np.array(grayscale)

    threshold = array.mean() - 20
    mask = array < threshold

    coords = np.column_stack(np.where(mask))
    if coords.shape[0] < 20:
        return image  # not enough signal to estimate an angle safely

    angle = _principal_axis_angle(coords)

    if abs(angle) < 0.5 or abs(angle) > 45:
        return image  # not worth rotating / likely a bad estimate

    return image.rotate(angle, expand=True, fillcolor=(255, 255, 255))


def _principal_axis_angle(coords):
    """Angle (degrees) of the best-fit line through a set of (row, col) points."""

    ys, xs = coords[:, 0], coords[:, 1]
    x_mean, y_mean = xs.mean(), ys.mean()
    cov_xy = np.mean((xs - x_mean) * (ys - y_mean))
    cov_xx = np.mean((xs - x_mean) ** 2)
    cov_yy = np.mean((ys - y_mean) ** 2)

    angle_rad = 0.5 * np.arctan2(2 * cov_xy, cov_xx - cov_yy)
    return np.degrees(angle_rad)


def _otsu_threshold(array):
    """Classic Otsu's method: pick the threshold that best separates
    text pixels from background by maximizing between-class variance."""

    hist, _ = np.histogram(array, bins=256, range=(0, 256))
    total = array.size
    sum_total = np.dot(np.arange(256), hist)

    sum_bg = 0.0
    weight_bg = 0.0
    best_variance = 0.0
    best_threshold = 128

    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        between_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between_variance > best_variance:
            best_variance = between_variance
            best_threshold = t

    return best_threshold


def binarize(image):
    """Otsu adaptive binarization - much gentler on thin text than a
    fixed threshold + blur, which tends to erode small font strokes."""

    grayscale = ImageOps.grayscale(image)
    array = np.array(grayscale)

    threshold = _otsu_threshold(array)
    binary = (array > threshold).astype(np.uint8) * 255

    return Image.fromarray(binary)


def preprocess_document(file_path, dpi=200):
    """Full pipeline: load -> deskew -> binarize, per page."""

    pages = load_document_as_images(file_path, dpi=dpi)
    processed = [binarize(deskew(page)) for page in pages]
    return processed


if __name__ == "__main__":
    from PIL import ImageDraw

    img = Image.new("RGB", (600, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 80), "INVOICE INV-2026-0913 Total: $95.50", fill="black")
    img.save("/tmp/sample.png")

    result = preprocess_document("/tmp/sample.png")
    print(f"Pages processed: {len(result)}")
    print(f"Output size: {result[0].size}, mode: {result[0].mode}")
    assert len(result) == 1
    assert result[0].mode == "L"
    print("ALL ASSERTIONS PASSED")
