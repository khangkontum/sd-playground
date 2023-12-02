import cv2


def create_mask(im: cv2.Mat, mask_ratio: float = 0.2) -> tuple[cv2.Mat, cv2.Mat]:
    cropped, mask = im.copy(), im.copy()

    cropped[0 : int(im.shape[0] * mask_ratio)] = 0
    cropped[-int(im.shape[0] * mask_ratio) :] = 0
    cropped[:, 0 : int(im.shape[1] * mask_ratio)] = 0
    cropped[:, -int(im.shape[1] * mask_ratio) :] = 0

    mask[:]=0
    mask[0 : int(im.shape[0] * mask_ratio)] = 255
    mask[-int(im.shape[0] * mask_ratio) :] = 255
    mask[:, 0 : int(im.shape[1] * mask_ratio)] = 255
    mask[:, -int(im.shape[1] * mask_ratio) :] = 255

    # cv2.imshow("cropped", cropped)
    # cv2.imshow("mask", mask)
    # cv2.imshow("original", im)
    # cv2.waitKey()
    return cropped, mask


if __name__ == "__main__":
    import sys

    # if len(sys.argv) != 4:
    #     print("Invalid number of argument, requires 3.")

    input_image_path = sys.argv[1]
    im = cv2.imread(input_image_path)
    cropped, mask = create_mask(im)
    # mask, resized = create_outpainting_mask(input_image_path)

    # Save the mask as an image
    cv2.imwrite(sys.argv[2], mask)
    cv2.imwrite(sys.argv[3], cropped)
