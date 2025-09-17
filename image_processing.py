import cv2
import numpy as np
from skimage.morphology import skeletonize
from matplotlib import pyplot as plt
import os

def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not load image from {path}")
    else:
        print("Image loaded successfully.")

    return img

def crop_padding_image(image):
    height, width = image.shape

    up, down, left, right = None, None, None, None

    for i in range(0,height):
        line = image[i][:]
        if 0 in line:
            up = i
            break

    for i in range(height-1,0,-1):
        line = image[i][:]
        if 0 in line:
            down = i
            break

    for j in range(0,width):
        line = image[:, j]
        if 0 in line:
            left = j
            break

    for j in range(width-1, 0, -1):
        line = image[:, j]
        if 0 in line:
            right = j
            break

    cropped_image = image[up : down, left: right]
    return cropped_image

def split_digits_grid(image, save_dir=None, label=None, show_plots=False, target_size=(28, 28)):
    """
    Ako je save_dir zadat, svaka obrađena pod-slika se snima u taj folder
    kao {label}_{i}_{j}.png. Ako je show_plots=True, prikazuje 1x2 subplot.
    """
    height, width = image.shape
    single_height = height // 10
    single_width = width // 12

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for i in range(0, 10):
        for j in range(0, 12):
            single_img = image[i*single_height:i*single_height + single_height,
                               j*single_width:j*single_width + single_width]

            processed_img = processing_pipeline(single_img, target_size=target_size)

            # snimi ako je traženo
            if save_dir is not None:
                fname = f"{label}_{i}_{j}.png" if label is not None else f"{i}_{j}.png"
                out_path = os.path.join(save_dir, fname)
                # processed_img = (processed_img > 0).astype(np.uint8) * 255
                cv2.imwrite(out_path, processed_img, [cv2.IMWRITE_PNG_COMPRESSION, 9])

            if show_plots:
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(single_img, cmap="gray")
                plt.title("Original")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                plt.imshow(processed_img, cmap="gray")
                plt.title("Processed")
                plt.axis("off")

                plt.tight_layout()
                plt.show()

def binary_mask(image, threshold=230):
    """Vrati ČISTO binarnu sliku: 0 ili 255 (ništa između)."""
    bin_img = (image > threshold).astype(np.uint8) * 255  # True->255, False->0
    return bin_img

def image_dilatation(image, kernel_dim = (3, 3), iterations_num=2):
    kernel = np.ones(kernel_dim , np.uint8)
    dilated_img = cv2.erode(image, kernel, iterations=iterations_num)
    return dilated_img

def image_erosion(image, kernel_dim = (3, 3), iterations_num=2):
    kernel = np.ones(kernel_dim , np.uint8)
    eroded_img = cv2.erode(255 - image, kernel, iterations=iterations_num)
    return 255 - eroded_img

def resize_image(image, target_size=(32, 32)):
    # NEAREST bez sivljenja + re-binarizacija na 0/1
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_NEAREST)
    # resized01 = (resized > 0).astype(np.uint8)  # 0/1 umesto 0/255
    return resized

def skeletonize_image(image):
    image = image < 150   # crno postaje True
    skeleton = skeletonize(image)
    # Pretvori nazad u uint8: True=linija=crno (0), False=pozadina=belo (255)
    skeleton_img = np.where(skeleton, 0, 255).astype(np.uint8)

    return skeleton_img

def processing_pipeline(image, target_size=(32, 32)):

    img = cv2.medianBlur(image, 5)
    img = binary_mask(img)
    img = image_dilatation(img,iterations_num=3)
    img = image_erosion(img,iterations_num=4)
    img = crop_padding_image(img)
    img = resize_image(img, target_size)
    # img = skeletonize_image(img)

    return img

if __name__ == "__main__":
    for d in range(10):
        image_path = f"data/{d}/cifra_{d}.jpg"
        raw_img = read_img(path=image_path)
        if raw_img is None:
            continue

        # Snimaj obrađene isečke u isti folder (data/d/)
        split_digits_grid(
            raw_img,
            save_dir=f"data/{d}",
            label=str(d),
            show_plots=False,
            target_size=(28, 28)
        )


