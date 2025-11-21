import numpy as np
import cv2
import os
from sklearn.cluster import KMeans

# Buat folder hasil
os.makedirs('hasil_segmentasi', exist_ok=True)

print("# SEGMENTASI 3 CITRA RGB")
print("Metode: Otsu + K-Means")

for i in range(1, 4):
    print(f"\nProses gambar {i}....")

    try:
        # 1. Baca gambar
        img = cv2.imread(f'gambar{i}.jpg')
        if img is None:
            print(f"Gambar gambar{i}.jpg tidak ditemukan, skip...")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. Otsu Thresholding
        _, otsu = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 4. K-Means Clustering
        pixels = img_rgb.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=5)
        labels = kmeans.fit_predict(pixels)
        kmeans_img = kmeans.cluster_centers_[labels].reshape(
            img_rgb.shape
        ).astype(np.uint8)

        # 5. Simpan hasil
        cv2.imwrite(f'hasil_segmentasi/otsu_{i}.jpg', otsu)
        cv2.imwrite(
            f'hasil_segmentasi/kmeans_{i}.jpg',
            cv2.cvtColor(kmeans_img, cv2.COLOR_RGB2BGR)
        )

        print(f"Gambar {i} selesai diproses")

    except Exception as e:
        print(f"Error processing gambar {i}: {e}")

print("\nSemua proses selesai! Hasil disimpan di folder 'hasil_segmentasi'")
