import torch, torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import umap

import matplotlib as mpl
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

def set_neurips_style():
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="DejaVu Sans",
    )

    mpl.rcParams.update({
        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # Fonts
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,

        # Axes
        "axes.linewidth": 0.8,
        "axes.labelweight": "normal",
        "axes.titleweight": "normal",

        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 4,

        # Ticks
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,

        # Legend
        "legend.frameon": False,

        # PDF/vector output
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Avoid transparent weirdness in some PDF viewers
        "savefig.transparent": False,
    })
set_neurips_style()


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

def load_cifar(batch_size=64, num_workers=12):
    transform = create_tform_cifar()
    datasets = {
        x: torchvision.datasets.CIFAR10(root='./cifar10-data', train=(x=="train"), download=True, transform=transform[x])
        for x in ["train", "val"]
    }
    return make_dataloaders(datasets, batch_size, num_workers)

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
    epochs = range(len(df[df.columns[0]]))
    
    for config in configs:
        sns.lineplot(df, x=epochs, y=config["column"], marker="o", **config["lineplot_kwargs"])
        config["lineplot_kwargs"]["ax"].set_title(config["title"])
        config["lineplot_kwargs"]["ax"].set_xlabel(config["xlabel"])
        config["lineplot_kwargs"]["ax"].set_ylabel(config["ylabel"])
        
        if "ylim" in config:
            config["lineplot_kwargs"]["ax"].set_ylim((config["ylim"]))

        config["lineplot_kwargs"]["ax"].xaxis.set_major_locator(MaxNLocator(integer=True))
        
        sns.despine(ax=config["lineplot_kwargs"]["ax"], left=False, bottom=False)

def plot_bar(values, categories, title=None, xlabel="", ylabel="", fmt="%.0f%%", figsize=(3.25, 2.4)):
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(x=categories, y=values, palette="colorblind", hue=categories, legend=False, ax=ax)
    for container in ax.containers:
        ax.bar_label(
            container,
            fmt=fmt,
            padding=2,
            fontsize=7,
            color="#333333",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.grid(axis="x", visible=False)

    sns.despine(ax=ax, left=False, bottom=False)
    fig.tight_layout()

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
    
    fig = plt.figure(figsize=(7.0,2.2))
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
            ax.set_title(
                f"{class_names[predicted_label].title()}\n"
                f"GT: {class_names[label].title()} "
                f"({confidence:.0%})",
                fontsize=8,
                fontweight="normal",
            )

            imshow(inputs.cpu(), process_fn)

            if images_so_far == num_images:
                model.train(mode=was_training)
                sns.despine(left=True, bottom=True)
                fig.subplots_adjust(
                    left=0,
                    right=1,
                    bottom=0,
                    top=0.82 if title is not None else 1,
                    wspace=0.03,
                )
                if title is not None:
                    plt.suptitle(title, fontweight="bold")
                return

    model.train(mode=was_training)
    sns.despine(left=True, bottom=True)
    fig.subplots_adjust(
        left=0,
        right=1,
        bottom=0,
        top=0.82 if title is not None else 1,
        wspace=0.03,
    )
    if title is not None:
        plt.suptitle(title, fontweight="bold")

def visualize_umap_features(features, labels, class_names, ax, legend=False, title=None):
    features = features.reshape((features.shape[0], -1))
    label_names = [class_names[x.numpy().tolist()] for x in labels]
    
    reducer = umap.UMAP(random_state=42, n_jobs=1)
    embedding = reducer.fit_transform(features)
    
    sns.scatterplot(
        data=pd.DataFrame({
            "umap_1": embedding[:,0],
            "umap_2": embedding[:,1],
            "label": label_names,
            }),
        x='umap_1',
        y='umap_2',
        hue='label',
        palette='colorblind',
        s=10,
        alpha=0.65,
        edgecolor='none',
        linewidth=0,
        legend=legend,
        ax=ax,
    )

    ax.set_xlabel('UMAP Dimension 1')
    ax.set_ylabel('UMAP Dimension 2')

    if title is not None:
        ax.set_title(title)

    ax.set_aspect("equal", "datalim")
    ax.grid(False)
    sns.despine(ax=ax)
