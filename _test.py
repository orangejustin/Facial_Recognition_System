import subprocess
import os
import torchvision
import torch
from models import CustomResNet
from tqdm import tqdm
from PIL import Image



class ClassificationTestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms):
        self.data_dir   = data_dir
        self.transforms = transforms

        # This one-liner basically generates a sorted list of full paths to each image in the test directory
        self.img_paths  = list(map(lambda fname: os.path.join(self.data_dir, fname), sorted(os.listdir(self.data_dir))))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return self.transforms(Image.open(self.img_paths[idx]))


def test(model, dataloader):
      model.eval()
      batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Test')
      test_results = []

      for i, (images) in enumerate(dataloader):
          # TODO: Finish predicting on the test set.
          images = images.to(DEVICE)

          with torch.inference_mode():
            outputs = model(images)

          outputs = torch.argmax(outputs, axis=1).detach().cpu().numpy().tolist()
          test_results.extend(outputs)

          batch_bar.update()

      batch_bar.close()
      return test_results

DATA_DIR    = 'content/data/11-785-f23-hw2p2-classification'
TEST_DIR    = os.path.join(DATA_DIR, "test")
DEVICE = 'cuda'
MODEL_DIR = './hw2_classification_2_more.pth'

checkpoint = torch.load('self_reset_exper_48_new.pth')
model = CustomResNet(dropout=True, dropout_prob=0.48).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
best_val_acc = checkpoint['val_acc']
train_acc = checkpoint['train_acc']
epoch = checkpoint['epoch']
print(f'Best validation accuracy is: {best_val_acc} at {epoch} epoch with training acc: {train_acc}')

valid_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

test_dataset = ClassificationTestDataset(TEST_DIR, transforms = valid_transforms)
#Why are we using val_transforms for Test Data?
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size = 128,
                                          shuffle = False,
                                          drop_last = False,
                                          num_workers = 4)
#
test_results = test(model, test_loader)

with open("./submission.csv", "w+") as f:
    f.write("id,label\n")
    for i in range(len(test_dataset)):
        f.write("{},{}\n".format(str(i).zfill(6) + ".jpg", test_results[i]))

cmd = "kaggle competitions submit -c 11-785-f23-hw2p2-classification -f submission.csv -m 'better :)'"
subprocess.run(cmd, shell=True, check=True)