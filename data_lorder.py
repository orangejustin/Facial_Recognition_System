import os

DATA_DIR    = 'content/data/11-785-f23-hw2p2-classification/'
TRAIN_DIR   = os.path.join(DATA_DIR, "train")
VAL_DIR     = os.path.join(DATA_DIR, "dev")
TEST_DIR    = os.path.join(DATA_DIR, "test")

def load():
    # Transforms using torchvision - Refer https://pytorch.org/vision/stable/transforms.html
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])# Implementing the right train transforms/augmentation methods is key to improving performance.

    # Most torchvision transforms are done on PIL images. So you convert it into a tensor at the end with ToTensor()
    # But there are some transforms which are performed after ToTensor() : e.g - Normalization
    # Normalization Tip - Do not blindly use normalization that is not suitable for this dataset

    valid_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])


    train_dataset   = torchvision.datasets.ImageFolder(TRAIN_DIR, transform= train_transforms)
    valid_dataset   = torchvision.datasets.ImageFolder(VAL_DIR, transform= valid_transforms)
    # You should NOT have data augmentation on the validation set. Why?


    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset     = train_dataset,
        batch_size  = config['batch_size'],
        shuffle     = True,
        num_workers = 7,
        pin_memory  = True
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset     = valid_dataset,
        batch_size  = config['batch_size'],
        shuffle     = False,
        num_workers = 6
    )
    print("Number of classes    : ", len(train_dataset.classes))
    print("No. of train images  : ", train_dataset.__len__())
    print("Shape of image       : ", train_dataset[0][0].shape)
    print("Batch size           : ", config['batch_size'])
    print("Train batches        : ", train_loader.__len__())
    print("Val batches          : ", valid_loader.__len__())

    return train_loader, valid_loader

# if __name__ == '__main__':

