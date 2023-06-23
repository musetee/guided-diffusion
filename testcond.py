from guided_diffusion.my_dataset import get_file_list
from guided_diffusion.image_datasets import _list_image_files_recursively
import os
import blobfile as bf

def get_cond(data_dir):
    all_files = _list_image_files_recursively(data_dir)
    class_names = [bf.basename(path).split("_")[0] for path in all_files]
    sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    classes = [sorted_classes[x] for x in class_names]
    print(classes)
    return classes


def get_image_basename(data_dir, train_number=10, val_number=1):
    train_ds, val_ds = get_file_list(data_dir, train_number, val_number)
    train_image_list = [i['image'] for i in train_ds]
    train_image_basename_list = [os.path.basename(os.path.dirname(i)) for i in train_image_list]
    print(train_image_basename_list)
    return train_image_basename_list

if __name__ == "__main__":
    data_dir = 'F:\yang_Projects\Datasets\Task1\pelvis'
    #_ = get_image_basename(data_dir, train_number=10, val_number=1)
    _ = get_cond(data_dir)
