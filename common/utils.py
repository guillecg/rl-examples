
def image_to_pytorch(img):
    return img.reshape(img.shape[-1], *img.shape[:-1])
