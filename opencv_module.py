import cv2
from PIL import Image
import tempfile

def process_image_for_rooftop(image_file):
    # Convert to OpenCV format
    image = Image.open(image_file).convert("RGB")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        image.save(temp_file.name)
        img = cv2.imread(temp_file.name)

    # Resize for consistent processing
    img = cv2.resize(img, (600, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Basic threshold to isolate flat surfaces
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    # Find contours of potential rooftop areas
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    area_px = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5000 < area < 50000:  # Filter small noise and huge objects
            area_px += area
            cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)

    return img, area_px
