import torch
from PIL import Image
import os
import argparse

import torchvision.transforms as transforms
import torch.nn as nn

from neuralNet.net import Siamese

parser = argparse.ArgumentParser(description="Scoring similarity between 2 images")

parser.add_argument('--root_dir', default='testSpace/')
parser.add_argument('--image1', required=True, help="Path to the first image")
parser.add_argument('--image2', required=True, help="Path to the second image")

args = parser.parse_args()


def main():
    root_dir = args.root_dir
    img1_path = args.image1
    img2_path = args.image2

    Siamese_network = Siamese()
    checkpoint = torch.load("model/checkpoint")
    Siamese_network.load_state_dict(checkpoint)
    Siamese_network.eval()

    with torch.no_grad():
        img1 = Image.open(os.path.join(root_dir, img1_path))
        img2 = Image.open(os.path.join(root_dir, img2_path))

        gray1 = img1.convert("L")
        gray2 = img2.convert("L")

        Tensor1 = transforms.ToTensor()(gray1).unsqueeze(0)
        Tensor2 = transforms.ToTensor()(gray2).unsqueeze(0)

        embedding1, embedding2 = Siamese_network(Tensor1, Tensor2)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cosine_similarity = cos(embedding1, embedding2).item()

        similarity = "similar" if cosine_similarity > 0.65 else "dissimilar"

        print("Cosine_Similarity : {}".format(cosine_similarity))
        print("Detected : {}".format(similarity))


if __name__ == '__main__':
    main()
