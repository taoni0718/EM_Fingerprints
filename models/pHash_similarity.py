import imagehash
from PIL import Image

def phash_similarity(image1, image2):
    phash1 = imagehash.phash(Image.fromarray(np.uint8(image1 * 255)))
    phash2 = imagehash.phash(Image.fromarray(np.uint8(image2 * 255)))

    return (phash1 - phash2) / len(phash1.hash)**2