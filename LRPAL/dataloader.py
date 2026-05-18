import os
import torch
import numpy as np
from torch.utils.data import Dataset, Subset, random_split, DataLoader
from torch_geometric.data import Data, Batch
from config import get_config
from torchvision import transforms

cfg = get_config()


def custom_collate(batch):

    graph_datas = [item for item in batch]
    images = [item.image for item in batch]
    pyg_batch = Batch.from_data_list(graph_datas)
    image_batch = torch.stack(images, dim=0)  # [B, 3, H, W]
    return pyg_batch, image_batch


class AnomalyDataset(Dataset):


    def __init__(self, data_dir, img_size=(300, 300)):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.resize = transforms.Resize(img_size)


        self.files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith(".pt") or f.endswith(".pth")
        ])

        if not self.files:
            raise FileNotFoundError(f"❌ 未在 {data_dir} 找到任何 .pt 文件")
        print(f"📁 加载 {len(self.files)} 个样本")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        data = torch.load(path, map_location="cpu")


        x = data.x.float()
        edge_index = data.edge_index.long()
        y = torch.tensor(int(data.y)).long()


        num_nodes = x.shape[0]
        if edge_index.numel() > 0:
            max_idx = edge_index.max().item()
            if max_idx == num_nodes:
                edge_index -= 1
            if edge_index.max().item() >= num_nodes or edge_index.min().item() < 0:
                raise ValueError(f"样本 {self.files[idx]} 的edge_index无效")


        img = data.image.float()
        if img.ndim == 2:
            img = img.unsqueeze(0)
        elif img.ndim == 3:
            img = img.permute(2, 0, 1).contiguous()
        else:
            raise ValueError(f"样本 {self.files[idx]} 图像维度异常（{img.ndim}D）")

  
        if img.shape[0] == 192:
            img = img.view(3, 64, img.shape[1], img.shape[2]).mean(dim=1)
        elif img.shape[0] != 3:
            raise ValueError(f"样本 {self.files[idx]} 通道数异常（{img.shape[0]}）")

        img = self.resize(img)

        return Data(x=x, edge_index=edge_index, y=y, image=img)

    def get_labels(self):
      
        labels = []
        for i in range(len(self.files)):
            path = os.path.join(self.data_dir, self.files[i])
            data = torch.load(path, map_location="cpu")
            labels.append(int(data.y))
        return np.array(labels)


def get_full_dataset(data_dir, img_size=(300, 300)):

    return AnomalyDataset(data_dir, img_size)


def get_dataset_labels(dataset):

    if hasattr(dataset, 'get_labels'):
        return dataset.get_labels()
    else:

        if isinstance(dataset, Subset):
            full_dataset = dataset.dataset
            indices = dataset.indices
            if hasattr(full_dataset, 'get_labels'):
                return full_dataset.get_labels()[indices]
            else:
   
                labels = []
                for i in indices:
                    data = full_dataset[i]
                    if hasattr(data, 'y'):
                        labels.append(data.y.item())
                    else:
        
                        labels.append(data[0].y.item())
                return np.array(labels)
        else:
       
            labels = []
            for i in range(len(dataset)):
                data = dataset[i]
                if hasattr(data, 'y'):
                    labels.append(data.y.item())
                else:
                  
                    labels.append(data[0].y.item())
            return np.array(labels)


def build_datasets(data_dir, val_ratio=0.2, init_label_ratio=0.1):

    dataset = AnomalyDataset(data_dir)
    total_len = len(dataset)
    val_len = int(total_len * val_ratio)
    train_len = total_len - val_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])


    train_labels = get_dataset_labels(train_dataset)


    init_label_len = max(1, int(train_len * init_label_ratio))


    from sklearn.model_selection import train_test_split
    indices = list(range(train_len))
    labeled_idx, unlabeled_idx = train_test_split(
        indices,
        test_size=1 - init_label_ratio,
        stratify=train_labels,
        random_state=42
    )

    Dl = Subset(train_dataset, labeled_idx)
    Du = Subset(train_dataset, unlabeled_idx)
    Val = val_dataset

    print(f"构建数据集: 总样本={total_len}, 训练集={train_len}, 验证集={val_len}")
    print(f"初始有标签={len(Dl)}, 无标签={len(Du)}")

    return Dl, Du, Val


def create_kfold_splits(full_dataset, k_folds=5, val_ratio=0.2, seed=42):

    all_labels = get_dataset_labels(full_dataset)
    n_samples = len(full_dataset)

    # 使用分层K折
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    splits = []

    for train_val_idx, test_idx in skf.split(range(n_samples), all_labels):

        train_val_labels = all_labels[train_val_idx]

      
        from sklearn.model_selection import train_test_split
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_ratio,
            stratify=train_val_labels,
            random_state=seed
        )


        if isinstance(train_idx, np.ndarray):
            train_idx = train_idx.tolist()
        if isinstance(val_idx, np.ndarray):
            val_idx = val_idx.tolist()
        if isinstance(test_idx, np.ndarray):
            test_idx = test_idx.tolist()

        splits.append((train_idx, val_idx, test_idx))

    print(f"创建了 {k_folds} 折交叉验证划分")
    for i, (train_idx, val_idx, test_idx) in enumerate(splits):
        print(f"  折 {i + 1}: 训练集={len(train_idx)}, 验证集={len(val_idx)}, 测试集={len(test_idx)}")

    return splits


def create_active_learning_splits(train_dataset, init_label_ratio=0.05, seed=42):
  
    if isinstance(train_dataset, Subset):
        indices = train_dataset.indices
        if isinstance(indices, np.ndarray):
            indices = indices.tolist()
        full_dataset = train_dataset.dataset
    else:
        indices = list(range(len(train_dataset)))
        full_dataset = train_dataset


    labels = get_dataset_labels(train_dataset)


    from sklearn.model_selection import train_test_split
    labeled_idx, unlabeled_idx = train_test_split(
        indices,
        test_size=1 - init_label_ratio,
        stratify=labels,
        random_state=seed
    )


    if isinstance(labeled_idx, np.ndarray):
        labeled_idx = labeled_idx.tolist()
    if isinstance(unlabeled_idx, np.ndarray):
        unlabeled_idx = unlabeled_idx.tolist()

    Dl = Subset(full_dataset, labeled_idx)
    Du = Subset(full_dataset, unlabeled_idx)

    print(f"主动学习划分: 总训练样本={len(indices)}, "
          f"有标签={len(Dl)}({len(Dl) / len(indices):.1%}), "
          f"无标签={len(Du)}({len(Du) / len(indices):.1%})")

    return Dl, Du


def get_dataloader(dataset, shuffle=True, batch_size=None):
  
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        collate_fn=custom_collate
    )


def update_datasets(Dl, Du, selected_indices):

    assert Dl.dataset is Du.dataset, 


    dl_indices = list(Dl.indices) if hasattr(Dl.indices, '__iter__') else [Dl.indices]
    du_indices = list(Du.indices) if hasattr(Du.indices, '__iter__') else [Du.indices]

    if isinstance(selected_indices, np.ndarray):
        selected_indices = selected_indices.tolist()

    selected_global = [du_indices[i] for i in selected_indices]


    dl_updated_indices = dl_indices + selected_global

    du_updated_indices = [i for i in du_indices if i not in selected_global]

    Dl_updated = Subset(Dl.dataset, dl_updated_indices)
    Du_updated = Subset(Du.dataset, du_updated_indices)

    return Dl_updated, Du_updated


def create_dataloaders(Dl, Du=None, Val=None, Test=None, batch_size=None):

    dataloaders = {}

    dataloaders['labeled'] = get_dataloader(Dl, shuffle=True, batch_size=batch_size)

    if Du is not None:
        dataloaders['unlabeled'] = get_dataloader(Du, shuffle=False, batch_size=batch_size)

    if Val is not None:
        dataloaders['val'] = get_dataloader(Val, shuffle=False, batch_size=batch_size)

    if Test is not None:
        dataloaders['test'] = get_dataloader(Test, shuffle=False, batch_size=batch_size)


    print("创建数据加载器:")
    for key, loader in dataloaders.items():
        dataset_size = len(loader.dataset)
        num_batches = len(loader)
        print(f"  {key}: 样本数={dataset_size}, 批次数={num_batches}")

    return dataloaders
