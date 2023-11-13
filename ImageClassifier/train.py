import argparse
import torch
from torch import nn, optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils import data
import os

def data_transformation(args):
    train_dir = os.path.join(args.data_directory, "train")
    valid_dir = os.path.join(args.data_directory, "valid")

    for folder in [args.data_directory, train_dir, valid_dir]:
        if not os.path.exists(folder):
            raise FileNotFoundError(f"Directory doesn't exist: {folder}")

    transform = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    data_loaders = {
        'train': data.DataLoader(ImageFolder(root=train_dir, transform=transform['train']), batch_size=32, shuffle=True),
        'valid': data.DataLoader(ImageFolder(root=valid_dir, transform=transform['valid']), batch_size=32, shuffle=True)
    }

    return data_loaders['train'], data_loaders['valid'], data_loaders['train'].dataset.class_to_idx

def save_checkpoint(model, optimizer, args, epoch, train_loss, valid_loss, accuracy, class_to_idx):
    checkpoint = {
        'vgg_type': args.model_arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'accuracy': accuracy
    }

    checkpoint_filename = 'checkpoint.pth'
    checkpoint_filepath = os.path.join(os.path.expanduser("~"), 'opt', checkpoint_filename)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
    
    torch.save(checkpoint, checkpoint_filepath)
    print(f'Checkpoint saved to {checkpoint_filepath}')

def train_model(args, train_data_loader, valid_data_loader, class_to_idx):
    model = getattr(models, args.model_arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(in_features=model.classifier[0].in_features, out_features=2048, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(in_features=2048, out_features=102, bias=True),
        nn.LogSoftmax(dim=1)
    )

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    model.to(device)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    print(f"Using {device} to train model.")

    for epoch in range(args.epochs):
        running_train_loss = sum(train_step(model, criterion, optimizer, device, inputs, labels) for inputs, labels in train_data_loader)
        average_train_loss = running_train_loss / len(train_data_loader)

        running_accuracy, running_valid_loss = 0, 0
        with torch.no_grad():
            for inputs, labels in valid_data_loader:
                valid_loss, accuracy = valid_step(model, criterion, device, inputs, labels)
                running_valid_loss += valid_loss.item()
                running_accuracy += accuracy

        average_valid_loss = running_valid_loss / len(valid_data_loader)
        accuracy = running_accuracy / len(valid_data_loader)

        print(f"Epoch: {epoch + 1}/{args.epochs} Train Loss: {average_train_loss:.3f} "
              f"Valid Loss: {average_valid_loss:.3f} Accuracy: {accuracy * 100:.3f}%")

        save_checkpoint(model, optimizer, args, epoch, average_train_loss, average_valid_loss, accuracy, class_to_idx)

def train_step(model, criterion, optimizer, device, inputs, labels):
    model.train()
    inputs, labels = inputs.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def valid_step(model, criterion, device, inputs, labels):
    model.eval()
    inputs, labels = inputs.to(device), labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    ps = torch.exp(outputs)
    top_p, top_class = ps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
    return loss, accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a neural network on images')
    parser.add_argument('data_directory', help='Path to the root data directory')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=0.003, help='Learning rate for training')
    parser.add_argument('--epochs', dest='epochs', type=int, default=5, help='Number of epochs for training')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--model_arch', dest='model_arch', type=str, default='vgg19',
                        choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'], help='Pre-trained model architecture')

    args = parser.parse_args()

    train_data_loader, valid_data_loader, class_to_idx = data_transformation(args)

    train_model(args, train_data_loader, valid_data_loader, class_to_idx)
