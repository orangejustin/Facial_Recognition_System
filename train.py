from tqdm.auto import tqdm
import torch


# Useful only in the case of compatible GPUs such as T4/V100=
_scaler = torch.cuda.amp.GradScaler()

def train(model, dataloader, config, optimizer, criterion, lr_scheduler, device, scaler=_scaler, use_cutmix=False):

    model.train()

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    num_correct = 0
    total_loss = 0

    for i, (images, labels) in enumerate(dataloader):

        optimizer.zero_grad()  # Zero gradients

        images, labels = images.to(device), labels.to(device)

        with torch.cuda.amp.autocast():  # This implements mixed precision.
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Update no. of correct predictions & loss as we iterate
        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (config['batch_size'] * (i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        )

        scaler.scale(loss).backward()  # This is a replacement for loss.backward()
        scaler.step(optimizer)  # This is a replacement for optimizer.step()
        scaler.update()

        if lr_scheduler is not None:
            lr_scheduler.step()

        batch_bar.update()  # Update tqdm bar

    batch_bar.close()  # You need this to close the tqdm bar

    acc = 100 * num_correct / (config['batch_size'] * len(dataloader))
    total_loss = float(total_loss / len(dataloader))

    return acc, total_loss


def train_cutmix(model, dataloader, config, optimizer, criterion, lr_scheduler, device, scaler=_scaler):

    model.train()

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    num_correct = 0
    total_loss = 0

    for i, (images, labels_onehot) in enumerate(dataloader):

        optimizer.zero_grad()  # Zero gradients

        images, labels_onehot = images.to(device), labels_onehot.to(device)

        with torch.cuda.amp.autocast():  # This implements mixed precision.
            outputs = model(x=images, labels=labels_onehot)
            loss = criterion(outputs, labels_onehot)

        # Update no. of correct predictions & loss as we iterate
        predicted_labels = torch.argmax(outputs, dim=1)
        true_labels = torch.argmax(labels_onehot, dim=1)
        num_correct += (predicted_labels == true_labels).sum().item()
        total_loss += float(loss.item())

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (config['batch_size'] * (i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        )

        scaler.scale(loss).backward()  # This is a replacement for loss.backward()
        scaler.step(optimizer)  # This is a replacement for optimizer.step()
        scaler.update()

        if lr_scheduler is not None:
            lr_scheduler.step()

        batch_bar.update()  # Update tqdm bar

    batch_bar.close()  # You need this to close the tqdm bar

    acc = 100 * num_correct / (config['batch_size'] * len(dataloader))
    total_loss = float(total_loss / len(dataloader))

    return acc, total_loss


def validate(model, dataloader, config, criterion, device):

    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val', ncols=5)

    num_correct = 0.0
    total_loss = 0.0

    for i, (images, labels) in enumerate(dataloader):

        # Move images to device
        images, labels = images.to(device), labels.to(device)

        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs, labels)

        num_correct += int((torch.argmax(outputs, axis=1) == labels).sum())
        total_loss += float(loss.item())

        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (config['batch_size']*(i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct)

        batch_bar.update()

    batch_bar.close()
    acc = 100 * num_correct / (config['batch_size']* len(dataloader))
    total_loss = float(total_loss / len(dataloader))
    return acc, total_loss