import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_6.jpg")

# prikazi originalnu sliku
# plt.figure()
# plt.title("Originalna slika")
# plt.imshow(img)
# plt.tight_layout()
# plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

unique_pixels = np.unique(img_array, axis=0)
print(len(np.unique(img_array, axis=0)))

# rezultatna slika
img_array_aprox = img_array.copy()
img_array_copy = img_array.copy()

n_clusters = 5

km = KMeans(n_clusters=n_clusters,n_init = 30, random_state = 0)

km.fit(img_array_aprox)

labels = km.predict(img_array_aprox)

centroids = km.cluster_centers_

img_array_aprox[:, 0] = centroids[labels][:, 0]
img_array_aprox[:, 1] = centroids[labels][:, 1]
img_array_aprox[:, 2] = centroids[labels][:, 2]

img_array_aprox = np.reshape(img_array_aprox,(w,h,d))

f,axarr = plt.subplots(1,2)

print(centroids) 

axarr[0].imshow(img)
axarr[1].imshow(img_array_aprox)
plt.tight_layout()
plt.show()

J_values = []
K_values = list(range(1,11))

for k in K_values:
    kmeans = KMeans(n_clusters=k,n_init = 5,random_state = 0)
    kmeans.fit(img_array_copy)
    J_values.append(kmeans.inertia_)

plt.figure()
plt.plot(K_values, J_values)
plt.title("Elbow metoda")
plt.xlabel("K")
plt.ylabel("J")
plt.grid(True)
plt.tight_layout()
plt.show()

for i in range(n_clusters):
    binary_mask = (labels == i).astype(np.uint8)
    binary_image = binary_mask.reshape((w, h))

    plt.figure()
    plt.title(f"Binarna slika {i + 1}")
    plt.imshow(binary_image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
