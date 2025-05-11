import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class SeamCarving:
    # CONSTRUTOR
    def __init__(self, image: np.ndarray):
        self.image = image
        self.height, self.width, _ = self.image.shape
        self.energy_map = self._calculate_energy()

    # 3.2.1
    def _calculate_energy(self):
        # Primeiro inicializamos a matriz de energia com zeros
        energy = np.zeros((self.height, self.width))

        for y in range(1, self.height - 1):
            for x in range(1, self.width - 1):
                # Calcula o gradiente em x
                dx_r = int(self.image[y, x + 1, 0]) - int(self.image[y, x - 1, 0])
                dx_g = int(self.image[y, x + 1, 1]) - int(self.image[y, x - 1, 1])
                dx_b = int(self.image[y, x + 1, 2]) - int(self.image[y, x - 1, 2])
                dx = dx_r**2 + dx_g**2 + dx_b**2

                # Calcula o gradiente em y
                dy_r = int(self.image[y + 1, x, 0]) - int(self.image[y - 1, x, 0])
                dy_g = int(self.image[y + 1, x, 1]) - int(self.image[y - 1, x, 1])
                dy_b = int(self.image[y + 1, x, 2]) - int(self.image[y - 1, x, 2])
                dy = dy_r**2 + dy_g**2 + dy_b**2

                # Calcula a energia
                energy[y, x] = np.sqrt(dx + dy)

        return energy

    # 3.3.2
    def find_vertical_seam(self):
        # Matriz para armazenar o custo acumulado
        dist_to = np.full((self.height, self.width), np.inf)

        # Matriz para armazenar o caminho (pixel anterior)
        edge_to = np.zeros((self.height, self.width), dtype=np.int32)

        # Inicializamos a primeira linha com os valores de energia
        dist_to[0] = self.energy_map[0]

        # Preenchemos as linhas seguintes
        for y in range(1, self.height):
            for x in range(self.width):
                # Pixel da esquerda e direita, bem como o próprio pixel
                for dx in [-1, 0, 1]:
                    prev_x = x + dx
                    if 0 <= prev_x < self.width:
                        if dist_to[y, x] > dist_to[y - 1, prev_x] + self.energy_map[y, x]:
                            dist_to[y, x] = dist_to[y - 1, prev_x] + self.energy_map[y, x]
                            edge_to[y, x] = prev_x

        # Encontra o pixel final com menor custo na última linha
        min_end = np.argmin(dist_to[-1])
        seam = []

        # Reconstrói o caminho de menor custo
        for y in range(self.height - 1, -1, -1):
            seam.append((y, min_end))
            min_end = edge_to[y, min_end]

        # Invertemos a ordem para obter o caminho na ordem correta
        seam.reverse()
        return seam

    # 3.4.1
    def remove_vertical_seam(self, seam):
        H, W, _ = self.image.shape
        new_image = np.zeros((H, W - 1, 3), dtype=self.image.dtype)

        for y, x in seam:
            # Dá delete ao pixel da seam na linha y
            new_image[y, :, :] = np.delete(self.image[y, :, :], x, axis=0)

        self.image = new_image
        self.height, self.width = new_image.shape[:2]
        self.energy_map = self._calculate_energy()

    def picture(self):
        return self.image.astype(np.uint8)
    
# Para testar o energy map
def get_energy_map(image):
    sc = SeamCarving(np.array(image))
    energy_map = sc.energy_map
    return energy_map

# 4.1
def resize_image(image, width_factor=None, height_factor=None):
    sc = SeamCarving(np.array(image))
    
    # Para a largura:
    if width_factor:
        new_width = int(sc.width * width_factor)
        while sc.width > new_width:
            seam = sc.find_vertical_seam()
            sc.remove_vertical_seam(seam)
    
    # Para a altura:
    if height_factor:
        new_height = int(sc.height * height_factor)
        sc.image = np.transpose(sc.image, (1, 0, 2))
        sc.height, sc.width = sc.width, sc.height
        sc.energy_map = sc._calculate_energy()
        while sc.width > new_height:
            seam = sc.find_vertical_seam()
            sc.remove_vertical_seam(seam)
        sc.image = np.transpose(sc.image, (1, 0, 2))
        sc.height, sc.width = sc.width, sc.height
        sc.energy_map = sc._calculate_energy()
    
    return sc.picture()


# E agora testamos tudo
imagem_1 = mpimg.imread("img-broadway_tower.jpg")
imagem_2 = mpimg.imread("img-brent-cox-unsplash.jpg")

energyMap_1 = get_energy_map(imagem_1)
energyMap_2 = get_energy_map(imagem_2)

nova_imagem = resize_image(imagem_1, width_factor=0.9)
#nova_imagem = resize_image(imagem_2, height_factor=0.6)

# O resultado:
plt.imshow(nova_imagem)
plt.show()