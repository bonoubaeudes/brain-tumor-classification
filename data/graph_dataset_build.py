import numpy as np
from skimage import segmentation, measure, feature
import cv2
import torch
from scipy.spatial.distance import pdist, squareform
from torch_geometric.data import Data

class ImageToGraphConverter:
    def __init__(self, num_segments=100, compactness=10):
        self.num_segments = num_segments
        self.compactness = compactness

    def extract_features_from_region(self, image, region_mask):
        region_pixels = image[region_mask]

        if len(region_pixels) == 0:
            return np.zeros(20)

        mean_intensity = np.mean(region_pixels)
        std_intensity = np.std(region_pixels)
        min_intensity = np.min(region_pixels)
        max_intensity = np.max(region_pixels)

        range_intensity = max_intensity - min_intensity
        cv_intensity = std_intensity / (mean_intensity + 1e-10)

        props = measure.regionprops(region_mask.astype(int))[0]
        area = props.area
        perimeter = props.perimeter
        eccentricity = props.eccentricity
        solidity = props.solidity

        #Entropy
        hist, _ = np.histogram(region_pixels, bins=256, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0

        #Caracteristic LBP
        try:
            image_uint8 = (image * 255).astype(np.uint8)
            lbp = feature.local_binary_pattern(
                image_uint8, 8, 1, method='uniform'
            )
            lbp_hist, _ = np.histogram(lbp[region_mask], bins=10, density=True)
        except Exception as e:
            print(f"Error LBP: {e}")
            lbp_hist = np.zeros(10)

        features = np.array([
            mean_intensity, std_intensity, min_intensity, max_intensity, area, perimeter, cv_intensity,
            entropy
        ] + lbp_hist.tolist())

        return features

    def build_graph_from_image(self, image_path, label, input_size=(224, 224)):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Impossible to load image: {image_path}")

        image = cv2.resize(image, input_size)

        img_min, img_max = image.min(), image.max()
        if img_max > img_min:
            image_normalized = (image - img_min) / (img_max - img_min)
        else:
            image_normalized = image / 255.0

        segments = segmentation.slic(
            image_normalized,
            n_segments = self.num_segments,
            compactness= self.compactness,
            start_label=1,
            channel_axis=None
        )

        node_features = []
        centroids = []
        for region_id in np.unique(segments):
            if region_id == 0:
                continue

            region_mask = (segments == region_id)

            features = self.extract_features_from_region(image_normalized, region_mask)
            node_features.append(features)

            try:
                props = measure.regionprops(region_mask.astype(int))[0]
                centroids.append([props.centroid[0], props.centroid[1]])
            except:
                continue

        if not node_features:
            raise ValueError(f"Invalid Node for image: {image_path}")

        node_features = np.array(node_features)
        centroids = np.array(centroids)

        # Construire les arêtes basées sur la distance
        if len(centroids) > 1:
            distances = squareform(pdist(centroids))
            k = min(5, len(centroids) - 1)
            edges = []

            for i in range(len(centroids)):
                nearest_indices = np.argsort(distances[i])[1:k+1]
                for j in nearest_indices:
                    edges.append([i, j])
                    edges.append([j, i])  # Graphe non dirigé

            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        # Créer le graphe PyTorch Geometric
        x = torch.tensor(node_features, dtype=torch.float)
        y = torch.tensor([label], dtype=torch.long)
        graph_data = Data(x=x, edge_index=edge_index, y=y)

        return graph_data, segments