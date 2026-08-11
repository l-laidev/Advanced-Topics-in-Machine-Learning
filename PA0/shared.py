import torch, torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import umap

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns
import pandas as pd

from tqdm.notebook import trange, tqdm
import os, time, json, gc
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()
DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


class GradientTracker:
    def __init__(self, model):
        self.gradients = {name: [] for name, _ in model.named_parameters()}
        self.handles = []
        self._register_hooks(model)
        
    def _register_hooks(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                def hook_fn(grad, p_name=name):
                    grad_norm = grad.norm().item()
                    grad_mean = grad.mean().item()
                    grad_std  = grad.std().item()
                    
                    self.gradients[p_name].append({
                        'norm': grad_norm,
                        'mean': grad_mean,
                        'std': grad_std
                    })
                    return grad
                
                handle = param.register_hook(hook_fn)
                self.handles.append(handle)
    
    def unregister(self):
        for handle in self.handles:
            handle.remove()

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


def make_dataloader(dataset, batch_size, n_workers=12):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                        num_workers=n_workers, pin_memory=(DEVICE!="cpu"), persistent_workers=(n_workers!=0))
    dataset_size = len(dataset)
    return dataloader, dataset_size

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

def plot_bar(values, categories, title, xlabel, ylabel, fmt="%.0f%%", figsize=(4,4)):
    sns.set_theme(style="white")
    plt.figure(figsize=figsize)

    ax = sns.barplot(x=categories, y=values, palette="mako", hue=categories, legend=False)

    for i in range(len(values)):
        ax.bar_label(ax.containers[i], fmt=fmt, padding=4, fontsize=11, color="#333333", weight="bold")

    plt.title(title, fontsize=16, pad=20, weight="bold", color="#222222")
    plt.xlabel(xlabel, fontsize=12, labelpad=10, color="#444444")
    plt.ylabel(ylabel, fontsize=12, labelpad=10, color="#444444")

    sns.despine(left=False, bottom=False)

    plt.tight_layout()
    plt.show()

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

def visualize_umap_features(features, labels, class_names, title):
    features = features.reshape((features.shape[0], -1))
    label_names = [class_names[x.numpy().tolist()] for x in labels]
    
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(features)
    
    sns.set_theme(style="white", palette="muted")
    plt.figure(figsize=(6, 4))
    sns.scatterplot(
        data=pd.DataFrame({
            "umap_1": embedding[:,0],
            "umap_2": embedding[:,1],
            "label": label_names,
            }),
        x='umap_1',
        y='umap_2',
        hue='label',
        palette='tab10',
        s=30,
        alpha=0.8,
        edgecolor='none'
    )

    plt.title(title, fontsize=14)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.gca().set_aspect('equal', 'datalim')
    sns.despine()
    plt.tight_layout()
    plt.show()