import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class SeamCarving:
    def __init__(self, image):
        self.image = image.astype(np.int32)
        self.H, self.W, _ = self.image.shape
        self.energy_map = self._calculate_energy()

    def _calculate_energy(self):
        energy = np.zeros((self.H, self.W))
        for y in range(1, self.H-1):
            for x in range(1, self.W-1):
                dx = np.sum((self.image[y, x+1] - self.image[y, x-1])**2)
                dy = np.sum((self.image[y+1, x] - self.image[y-1, x])**2)
                energy[y, x] = np.sqrt(dx + dy)
        return energy

    def find_vertical_steam(self):
        H, W = self.H, self.W
        energy = self.energy_map
        dist_to = np.full((H, W), np.inf)
        edge_to = np.zeros((H, W), dtype=np.int32)

        dist_to[0] = energy[0]
        for y in range(1, H):
            for x in range(W):
                for dx in [-1, 0, 1]:
                    prev_x = x + dx
                    if 0 <= prev_x < W:
                        if dist_to[y, x] > dist_to[y-1, prev_x] + energy[y, x]:
                            dist_to[y, x] = dist_to[y-1, prev_x] + energy[y, x]
                            edge_to[y, x] = prev_x

        min_end = np.argmin(dist_to[-1])
        seam = []
        for y in range(H-1, -1, -1):
            seam.append((y, min_end))
            min_end = edge_to[y, min_end]
        seam.reverse()
        return seam

    def remove_vertical_steam(self, seam):
        new_image = np.zeros((self.H, self.W - 1, 3), dtype=np.int32)
        for y, x in seam:
            new_image[y, :, :] = np.delete(self.image[y, :, :], x, axis=0)
        self.image = new_image
        self.H, self.W = new_image.shape[:2]
        self.energy_map = self._calculate_energy()

    def picture(self):
        return self.image.astype(np.uint8)


def resize_image_seam_carving(image, scale_w=None, scale_h=None):
    sc = SeamCarving(np.array(image))
    if scale_w:
        new_width = int(sc.W * scale_w)
        while sc.W > new_width:
            seam = sc.find_vertical_steam()
            sc.remove_vertical_steam(seam)
    # Implementar caso scale_h para costura horizontal (transpor a imagem)
    return Image.fromarray(sc.picture())

from PIL import Image

img1 = Image.open("img-broadway_tower.jpg")
resized_img1 = resize_image_seam_carving(img1, scale_w=0.7)
plt.imshow(resized_img1)
plt.title("Reduzida para 70% da largura original")
plt.axis("off")
plt.show()

img2 = Image.open("img-brent-cox-unsplash.jpg")
resized_img2 = resize_image_seam_carving(img2.transpose(Image.TRANSPOSE), scale_w=0.6)
resized_img2 = resized_img2.transpose(Image.TRANSPOSE)
plt.imshow(resized_img2)
plt.title("Reduzida para 60% da altura original")
plt.axis("off")
plt.show()
