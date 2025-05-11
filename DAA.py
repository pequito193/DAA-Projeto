import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import PIL as pillow

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

        # Os pixeis fronteiriços estavam constantemente a causar problemas, então decidimos dar-lhes um valor fixo elevado para não serem escolhidos
        energy[0, :] = 1000
        energy[-1, :] = 1000
        energy[:, 0] = 1000
        energy[:, -1] = 1000

        # 3.2.2 - Análise da complexidade
        # A complexidade em relação ao tempo é O(altura*largura), porque percorremos cada pixel da imagem e calculamos o seu gradiente.
        # Podemos concluir que a complexidade é linear em relação ao número de pixels da imagem.
        # A complexidade em relação ao espaço é O(altura*largura), porque armazenamos a matriz de energia com o mesmo tamanho da imagem original.
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
        cost = np.copy(self.energy_map)
        path = np.zeros((self.height, self.width), dtype=np.int32)
        
        for y in range(1, self.height):
            for x in range(self.width):
                # Para cada pixel, verificamos os 3 pixeis acima dele
                if x == 0:
                    candidates = cost[y-1, 0:2]
                    min = np.argmin(candidates)
                    cost[y, x] += candidates[min]
                    path[y, x] = min
                elif x == self.width - 1:
                    candidates = cost[y-1, x-1:x+1]
                    min = np.argmin(candidates)
                    cost[y, x] += candidates[min]
                    path[y, x] = x - 1 + min
                else:
                    candidates = cost[y-1, x-1:x+2]
                    min = np.argmin(candidates)
                    cost[y, x] += candidates[min]
                    path[y, x] = x - 1 + min
        
        # Vamos a procura do caminho de menor energia
        seam = np.zeros(self.height, dtype=np.int32)
        seam[-1] = np.argmin(cost[-1, :])
        
        # Finalmente, fazemos o backtracking
        for y in range(self.height - 1, 0, -1):
            seam[y-1] = path[y, seam[y]]
            
        return seam

    # 3.4.1
    def remove_vertical_seam(self, seam):
        # Criamos uma nova imagem com um pixel a menos de largura (largura porque quando queremos reduzir a altura, rodamos a imagem 90 graus)
        new_image = np.zeros((self.height, self.width - 1, 3), dtype=self.image.dtype)
        
        # Copiamos a imagem original menos a seam que queremos remover
        for y in range(self.height):
            # Pixeis antes da seam
            new_image[y, 0:seam[y], :] = self.image[y, 0:seam[y], :]
            # Pixeis depois da seam
            new_image[y, seam[y]:, :] = self.image[y, seam[y]+1:, :]
        
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

nova_imagem_1 = resize_image(imagem_1, width_factor=0.7)
nova_imagem_2 = resize_image(imagem_2, height_factor=0.6)

# O resultado:
plt.imshow(nova_imagem_1)
plt.show()

plt.imshow(nova_imagem_2)
plt.show()