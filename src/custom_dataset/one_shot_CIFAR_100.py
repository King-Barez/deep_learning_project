import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import numpy as np
import random
from PIL import Image



class SiameseCIFAR100(Dataset):
    def __init__(self, root, train=True, download=False, transform=None, classes_to_include=None, same_class_ratio=0.5):
        super(SiameseCIFAR100, self).__init__()
        # Scarica il dataset CIFAR-100 e combina i set di training e test
        self.data = np.concatenate([datasets.CIFAR100(root, train=True, download=download).data, datasets.CIFAR100(root, train=False, download=download).data])
        self.targets = np.concatenate([datasets.CIFAR100(root, train=True, download=download).targets, datasets.CIFAR100(root, train=False, download=download).targets])

        # Inizializza i parametri del dataset, inclusi la trasformazione delle immagini e le classi da includere
        self.transform = transform
        self.train = train
        self.classes_to_include = classes_to_include if classes_to_include else list(range(100))
        self.same_class_ratio = same_class_ratio

        # Filtra il dataset per le classi selezionate
        self.data, self.targets = self.filter_dataset_by_classes()

        if self.train:
            self.group_examples()
        else:
            self.create_support_and_query_sets()

    def filter_dataset_by_classes(self):
        data = []
        targets = []
        for idx, target in enumerate(self.targets):
            if target in self.classes_to_include:
                data.append(self.data[idx])
                targets.append(target)
        return np.array(data), np.array(targets)
    
    def group_examples(self):
        np_arr = np.array(self.targets)
        self.grouped_examples = {i: np.where(np_arr == i)[0] for i in self.classes_to_include}

    def create_support_and_query_sets(self):
        self.support_set = {}
        self.query_set = []

        for class_id in self.classes_to_include:
            class_indices = np.where(self.targets == class_id)[0]

            # Seleziona una sola immagine casuale come supporto
            support_index = random.choice(class_indices)
            self.support_set[class_id] = self.data[support_index]

            # Escludi l'immagine di supporto dal query set
            remaining_indices = [idx for idx in class_indices if idx != support_index]

            # Aggiungi tutte le immagini rimanenti al query set
            for idx in remaining_indices:
                self.query_set.append((self.data[idx], class_id))

        print(f"Support set size: {len(self.support_set)}")
        print(f"Query set size: {len(self.query_set)}")

    def __len__(self):
        if self.train:
            return len(self.targets)
        else:
            return len(self.query_set)

    def __getitem__(self, index):
        if self.train:
            if random.random() < self.same_class_ratio:
                # Crea una coppia di immagini della stessa classe
                selected_class = random.choice(self.classes_to_include)
                random_index_1 = random.choice(self.grouped_examples[selected_class])
                image_1 = self.data[random_index_1]

                random_index_2 = random.choice(self.grouped_examples[selected_class])
                while random_index_2 == random_index_1:
                    random_index_2 = random.choice(self.grouped_examples[selected_class])
                image_2 = self.data[random_index_2]
                target = torch.tensor(1, dtype=torch.float)
            else:
                # Crea una coppia di immagini di classi diverse
                selected_class = random.choice(self.classes_to_include)
                random_index_1 = random.choice(self.grouped_examples[selected_class])
                image_1 = self.data[random_index_1]

                other_selected_class = random.choice(self.classes_to_include)
                while other_selected_class == selected_class:
                    other_selected_class = random.choice(self.classes_to_include)
                random_index_2 = random.choice(self.grouped_examples[other_selected_class])
                image_2 = self.data[random_index_2]
                target = torch.tensor(0, dtype=torch.float)

            image_1 = Image.fromarray(image_1)  # Converti in immagine PIL
            image_2 = Image.fromarray(image_2)  # Converti in immagine PIL

            if self.transform:
                image_1 = self.transform(image_1)
                image_2 = self.transform(image_2)

            return image_1, image_2, target

        else:
            # In fase di test restituisci solo l'immagine di query e la classe target
            query_image, query_class = self.query_set[index]
            query_image = Image.fromarray(query_image)  # Converti in immagine PIL

            if self.transform:
                query_image = self.transform(query_image)

            return query_image, torch.tensor(query_class, dtype=torch.long)