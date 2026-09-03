import cv2
import numpy as np

def preprocess(img):
    """
    Converts to grayscale and normalizes intensity.
    Input: BGR or grayscale image (numpy array)
    Output: float32 grayscale image, zero-mean, unit-std
    """
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    gray = gray.astype(np.float32)

    mean = gray.mean()
    std = gray.std()
    if std < 1e-6:
        std = 1e-6  # avoid divide by zero on blank patches

    normalized = (gray - mean) / std
    return normalized


if __name__ == "__main__":
    img = cv2.imread("../tests/synthetic_data/pair_0_image.png")
    processed = preprocess(img)
    print("Original shape:", img.shape)
    print("Processed shape:", processed.shape)
    print("Processed mean (should be ~0):", processed.mean())
    print("Processed std (should be ~1):", processed.std())