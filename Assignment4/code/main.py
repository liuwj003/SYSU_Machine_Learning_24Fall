import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from model import VAE
from argument import args
import os
from torchvision.utils import make_grid, save_image


def VAE_loss(x, recon_x, mean, log_var):
    """
    loss function to be minimized, loss = -ELBO
    :param x: input sample x
    :param recon_x: reconstructed x
    :param mean:
    :param log_var:
    :return: loss
    """
    # BCELoss函数的前 1 个参数是预测值，后 1 个参数是样本真实值
    # reduction不能改成 mean，否则反而降低了这个recon_loss在计算公式中所占的权重，最后会导致特征丢失
    recon_loss = nn.BCELoss(reduction='sum')(recon_x, x)
    # use ".pow(2)" ，按元素求幂
    kl_div = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    loss = recon_loss + kl_div
    return loss


def train(model, data_loader, optimizer, epochs, device):
    """
    train VAE model
    :param model:
    :param data_loader:
    :param optimizer:
    :param epochs:
    :param device:
    :return:
    """
    model.train()
    loss_history = []

    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (x_batch, _) in enumerate(data_loader):
            x_batch = x_batch.to(device)

            # forward
            recon_x, mean, log_var = model(x_batch)

            # compute loss
            loss = VAE_loss(x_batch, recon_x, mean, log_var)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 最后1个batch时，可视化重构的x
            if batch_idx == len(data_loader) - 1:
                visual_recon_x(model, x_batch, recon_x, epoch)

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx}/{len(data_loader)}], Loss: {loss.item():.4f}")

        avg_loss = train_loss / len(data_loader.dataset)
        loss_history.append(avg_loss)
        print(f"Average Loss: {avg_loss:.4f}")

    return loss_history


def plot_loss(loss_history, save_path='avg_loss_per_epoch.png'):
    plt.figure(figsize=(12, 8))
    plt.plot(loss_history, label='Average Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Average Loss per Epoch')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.show()


def visual_recon_x(model, x_batch, recon_x, epoch, save_dir='./recon_x'):
    """
    Visualize reconstructed samples
    """
    os.makedirs(save_dir, exist_ok=True)

    # x_batch = x_batch.cpu()
    # recon_x = recon_x.cpu()

    # grid: 原样本+重构样本
    combined = torch.cat([x_batch, recon_x], dim=0)
    grid = make_grid(combined, nrow=x_batch.size(0), normalize=True, value_range=(0, 1))

    # save png
    save_path = f"{save_dir}/epoch_{epoch + 1}.png"
    save_image(grid, save_path)


def generate_img(model, num, latent_dim, device, save_path='generated.png'):
    """
    Generate new images
    :param model: VAE
    :param num: number of images to generate
    :param latent_dim:
    :param device:
    :param save_path:
    :return:
    """
    model.eval()
    with torch.no_grad():
        # z ~ N(0, I)
        z = torch.randn(num, latent_dim).to(device)
        generated_img = model.generate(z)

    # plot images
    grid_size = int(num ** 0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < num:
            img = generated_img[i].permute(1, 2, 0).numpy()  # 把CHW格式变成HWC格式，也就是把通道数放到最后来
            ax.imshow(img, cmap='gray' if img.shape[2] == 1 else None)  # 灰度图像 or RGB
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Generated images saved to {save_path}")
    plt.show()


def get_single_class_cifar10(class_label, batch_size=64):
    """
    only get one class Data in CIFAR-10
    :return : DataLoader
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    full_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    indices = [idx for idx, (_, label) in enumerate(full_dataset) if label == class_label]
    subset = Subset(full_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


def main():
    arguments = args()
    learning_rate = arguments.learning_rate
    batch_size = arguments.batch_size
    epochs = arguments.epochs
    latent_dim = arguments.latent_dim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # preprocess data
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    # train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = get_single_class_cifar10(class_label=5, batch_size=batch_size)

    model = VAE(image_channels=3, latent_dim=latent_dim, nn_type='Conv').to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_history = train(model, train_loader, optimizer, epochs, device)

    generate_img(model=model, num=30, latent_dim=latent_dim, device=device)

    plot_loss(loss_history)


if __name__ == "__main__":
    main()




