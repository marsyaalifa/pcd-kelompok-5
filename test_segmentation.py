import unittest
import numpy as np
import cv2
import os
from sklearn.cluster import KMeans

class TestSegmentation(unittest.TestCase):
    
    def test_imports(self):
        """Test bahwa semua library bisa diimport"""
        try:
            import numpy as np
            import cv2
            from sklearn.cluster import KMeans
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import error: {e}")
    
    def test_otsu_threshold(self):
        """Test fungsi Otsu thresholding"""
        # Buat gambar dummy
        dummy_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        
        # Test Otsu thresholding
        ret, thresh = cv2.threshold(dummy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        self.assertIsInstance(ret, float)
        self.assertEqual(thresh.shape, (50, 50))
    
    def test_kmeans(self):
        """Test K-Means clustering"""
        # Data dummy
        dummy_pixels = np.random.rand(50, 3)
        
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=5)
        labels = kmeans.fit_predict(dummy_pixels)
        
        self.assertEqual(len(labels), 50)
        self.assertEqual(len(kmeans.cluster_centers_), 2)
    
    def test_folder_creation(self):
        """Test pembuatan folder hasil"""
        os.makedirs('test_folder', exist_ok=True)
        self.assertTrue(os.path.exists('test_folder'))
        # Bersihkan
        import shutil
        shutil.rmtree('test_folder')

if __name__ == '__main__':
    unittest.main()