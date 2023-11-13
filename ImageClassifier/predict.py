import argparse
import torch
from torchvision import transforms
from PIL import Image
import json
import torchvision.models as models

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    vgg_type = checkpoint['vgg_type']
    
    if vgg_type == "vgg11":
        model = models.vgg11(pretrained=True)
    elif vgg_type == "vgg13":
        model = models.vgg13(pretrained=True)
    elif vgg_type == "vgg16":
        model = models.vgg16(pretrained=True)
    elif vgg_type == "vgg19":
        model = models.vgg19(pretrained=True)
    else:
        print("Unsupported architecture. Please use 'vgg11', 'vgg13', 'vgg16', or 'vgg19'.")
        return None

    # Freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def process_image(image_path):
    pil_image = Image.open(image_path)
    
    # Resize and center crop the image
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply the transformations
    image = image_transform(pil_image)
    
    return image

def predict(image_path, model, topk, device, cat_to_name):
    model.to(device)
    model.eval()

    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)

    ps = torch.exp(output)
    top_p, top_classes = ps.topk(topk, dim=1)

    # Convert indices to class labels
    idx_to_flower = {class_idx: cat_to_name[str(class_idx + 1)] for class_idx in top_classes[0].tolist()}

    return top_p[0].tolist(), [idx_to_flower[idx] for idx in top_classes[0].tolist()]

def print_predictions(args, checkpoint_path):
    # Load model
    model = load_checkpoint(checkpoint_path)
    
    if model is None:
        return

    # Decide device depending on user arguments and device availability
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    with open(args.category_names_json_filepath, 'r') as f:
        cat_to_name = json.load(f)

    # Predict image
    top_ps, top_classes = predict(args.image_filepath, model, args.top_k, device, cat_to_name)

    print("Predictions:")
    for i in range(args.top_k):
        print("#{: <3} {: <25} Prob: {:.2f}%".format(i, top_classes[i], top_ps[i] * 100))

if __name__ == '__main__':
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument(dest='image_filepath', help="Path to the image file that you want to classify")

    # Optional arguments
    parser.add_argument('--category_names_json_filepath', dest='category_names_json_filepath',
                        help="Path to the JSON file that maps categories to real names", default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k', help="Number of most likely classes to return, default is 5", default=5, type=int)
    parser.add_argument('--gpu', dest='gpu', help="Include this argument if you want to use the GPU for inference", action='store_true')

    # Parse and print the results
    args = parser.parse_args()

    # Directly use the saved checkpoint file path
    checkpoint_path = "/root/opt/checkpoint.pth"

    print_predictions(args, checkpoint_path)
