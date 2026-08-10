import torch, torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns

from tqdm.notebook import trange, tqdm
import os, time, json, gc
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


def free_mem():
    gc.collect()
    torch.cuda.empty_cache()

def create_tform_cifar():
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

def make_dataloaders(datasets, batch_size, n_workers=12):
    dataloaders = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size, shuffle=True,
                                        num_workers=n_workers, pin_memory=(DEVICE!="cpu"), persistent_workers=(n_workers!=0))
        for x in ['train', 'val']
    }
    dataset_sizes = {
        x: len(datasets[x])
        for x in ['train', 'val']
    }
    return dataloaders, dataset_sizes

def load_cifar(batch_size=64):
    transform = create_tform_cifar()
    datasets = {
        x: torchvision.datasets.CIFAR10(root='./cifar10-data', train=(x=="train"), download=True, transform=transform[x])
        for x in ["train", "val"]
    }
    return make_dataloaders(datasets, batch_size)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def create_clf_head(n_features):
    return nn.Linear(n_features, 10, bias=True)

def create_classification(model, lr=0.001, momentum=0.9, step_size=7, gamma=0.1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    exp_lr_sched = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    return criterion, optimizer, exp_lr_sched

def train_model(fname, model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()
    metrics = {
        "train": {"loss": [], "accuracy": []},
        "val": {"loss": [], "accuracy": []},
    }

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, f'best_{fname}.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in (t := trange(num_epochs)):
            t.set_description_str(f'Epoch {epoch:3}/{num_epochs - 1:3}')

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in tqdm(dataloaders[phase], total=len(dataloaders[phase]), leave=False):
                    inputs = inputs.to(DEVICE, non_blocking=True)
                    labels = labels.to(DEVICE, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.detach() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss.item() / dataset_sizes[phase]
                epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

                t.set_description_str(f'[{epoch:3}] {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                metrics[phase]["loss"].append(epoch_loss)
                metrics[phase]["accuracy"].append(epoch_acc)

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    
    with open(f"{fname}.json", "w") as f:
        f.write(json.dumps(metrics))

def plot_line(df, configs):
    sns.set_theme(style="whitegrid")
    epochs = range(len(df[df.columns[0]]))
    
    for config in configs:
        sns.lineplot(df, x=epochs, y=config["column"], marker="o", **config["lineplot_kwargs"])
        config["lineplot_kwargs"]["ax"].set_title(config["title"], fontsize=18, pad=20, weight="bold")
        config["lineplot_kwargs"]["ax"].set_xlabel(config["xlabel"], fontsize=14, labelpad=10)
        config["lineplot_kwargs"]["ax"].set_ylabel(config["ylabel"], fontsize=14, labelpad=10)
        
        if "ylim" in config:
            config["lineplot_kwargs"]["ax"].set_ylim((config["ylim"]))

        config["lineplot_kwargs"]["ax"].xaxis.set_major_locator(MaxNLocator(integer=True))
        
    sns.despine(left=True, bottom=True)


def imshow(inp, process_fn, title=None):
    inp = inp.numpy()
    if len(inp.shape) == 3:
        inp = inp.transpose((1, 2, 0))
    
    inp = process_fn(inp)
    inp = np.clip(inp, 0, 1)
    
    plt.imshow(inp)
    plt.axis("off")
    if title is not None:
        plt.title(title, fontweight="bold")

def visualize_model(model, process_fn, samples, class_names, run_inference, title=None):
    was_training = model.training
    model.eval()
    images_so_far = 0
    num_images = len(samples)
    
    fig = plt.figure(figsize=(11,5))
    sns.set_theme(style='white', palette='deep')

    with torch.no_grad(), tqdm(total=num_images) as pbar:
        for i, (image, label) in enumerate(samples):
            inputs, predicted_label, confidence = run_inference(model, image)
            
            images_so_far += 1
            pbar.update(1)
            
            ax = plt.subplot(1, num_images, images_so_far)
            ax.axis('off')
            ax.set_title(f'Predicted: {class_names[predicted_label].title()}'
                            f'\nActual: {label}'
                            f'\n({int(confidence*100)}% Confidence)', fontweight="bold")
            imshow(inputs, process_fn)

            if images_so_far == num_images:
                model.train(mode=was_training)
                sns.despine(left=True, bottom=True)
                plt.tight_layout()
                if title is not None:
                    plt.suptitle(title, fontweight="bold")
                plt.tight_layout(rect=[0, 0, 1, 0.93])
                plt.show()
                return
        model.train(mode=was_training)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    if title is not None:
        plt.suptitle(title, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()