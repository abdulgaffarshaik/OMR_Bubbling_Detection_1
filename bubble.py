import cv2
import numpy as np
import os
import pandas as pd

# ---------------- CONFIG ----------------
INPUT_DIR = r"A:\omr_bubbling\some\bropped"  # folder containing all images
OUTPUT_CSV = r"A:\omr_bubbling\some\omr_results.csv"

# Cropping coordinates
MARKS_CROP = {"y1": 207, "y2": 520, "x1": 930, "x2": 1017}   # Total Marks #207 top 520 bottom 930 right 1017 left
SERIAL_CROP = {"y1": 207, "y2": 520, "x1": 1015, "x2": 1108} # Serial No.   #207 top 520 bottom 1015 right 1108 left

# Enhancement parameters
ENHANCE_BRIGHTNESS = 9
ENHANCE_CONTRAST = 0.9
GAMMA = 0.5

# Detection tuning
THRESHOLD_FILL_RATIO = 0.25
SHOW_DEBUG = False

# ---------------- IMAGE ENHANCEMENT ----------------
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def crop_and_enhance(img, crop):
    cropped = img[crop["y1"]:crop["y2"], crop["x1"]:crop["x2"]]
    yuv = cv2.cvtColor(cropped, cv2.COLOR_BGR2YUV)
    y = yuv[:,:,0]
    y = cv2.convertScaleAbs(y, alpha=ENHANCE_CONTRAST, beta=ENHANCE_BRIGHTNESS)
    yuv[:,:,0] = y
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    enhanced = adjust_gamma(enhanced, GAMMA)
    enhanced = cv2.GaussianBlur(enhanced, (3,3), 0)
    return enhanced

# ---------------- BUBBLE DETECTION ----------------
def detect_bubbles_and_numbers(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)
        if 10 < w < 40 and 10 < h < 40 and 0.7 < aspect_ratio < 1.3 and 100 < area < 1200:
            mask = np.zeros(thresh.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            filled_ratio = cv2.countNonZero(mask[y:y+h, x:x+w]) / (w*h)
            if filled_ratio > THRESHOLD_FILL_RATIO:
                bubbles.append((x, y, w, h))

    # Sort top-to-bottom
    bubbles = sorted(bubbles, key=lambda b: b[1])

    height = img.shape[0]
    section_height = height / 10
    filled_digits = []

    for (x, y, w, h) in bubbles:
        digit = int((y + h/2) // section_height)
        if 0 <= digit <= 9:
            filled_digits.append(digit)

    # Remove duplicates and sort
    filled_digits = sorted(list(set(filled_digits)))
    return filled_digits

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Get all image files
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    print(f"[INFO] Found {len(image_files)} image(s) in {INPUT_DIR}")

    results = []

    for filename in image_files:
        img_path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Could not read {filename}")
            continue

        # Crop & enhance in memory
        marks_img = crop_and_enhance(img, MARKS_CROP)
        serial_img = crop_and_enhance(img, SERIAL_CROP)

        # Detect bubbles
        marks_digits = detect_bubbles_and_numbers(marks_img)
        serial_digits = detect_bubbles_and_numbers(serial_img)

        total_marks_str = "".join(map(str, marks_digits)) if marks_digits else "None"
        serial_no_str = "".join(map(str, serial_digits)) if serial_digits else "None"

        print(f"[INFO] {filename} -> Serial: {serial_no_str}, Marks: {total_marks_str}")

        results.append({
            "Serial Number": serial_no_str,
            "Total Marks": total_marks_str,
            "Image": img_path
        })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[INFO] All results saved to {OUTPUT_CSV}")