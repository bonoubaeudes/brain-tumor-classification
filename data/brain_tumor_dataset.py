import os

from data.graph_dataset_build import ImageToGraphConverter


class BrainTumorGraphDataset:

    def __init__(self, data_dir, classes_names, transform=None, max_samples_per_class=None):
        self.data_dir = data_dir
        self.transform = transform
        self.max_samples_per_class = max_samples_per_class

        self.classes = classes_names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                class_samples = []
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(class_dir, img_name)
                        class_samples.append((img_path, self.class_to_idx[class_name]))


                if self.max_samples_per_class is not None:
                    class_samples = class_samples[:self.max_samples_per_class]
                    print(f"Class {class_name}: {len(class_samples)} sample ")
                else:
                    print(f"Class {class_name}: {len(class_samples)} samples")

                self.samples.extend(class_samples)

    def __len__(self):
        return len(self.samples)

def build_brain_tumor_graph(cfg, num_segments=100, compactness=10, max_samples_per_class=None):
    train_dir = cfg["data"]["train_dir"]
    test_dir = cfg["data"]["test_dir"]

    train_dataset = BrainTumorGraphDataset(train_dir, max_samples_per_class)
    test_dataset = BrainTumorGraphDataset(test_dir, max_samples_per_class)
    converter = ImageToGraphConverter(num_segments=100, compactness=10)
    graph_dataset_train = [converter.build_graph_from_image(img, lbl) for img, lbl in train_dataset.samples]
    graph_dataset_test = [converter.build_graph_from_image(img, lbl) for img, lbl in test_dataset.samples]
    print(f"Train Dataset  : {len(graph_dataset_train)} graph")
    print(f"Test Dataset : {len(graph_dataset_test)} graph")