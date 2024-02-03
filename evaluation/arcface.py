import torch
import torch.nn.functional as F


class ArcFaceModel(torch.nn.Module):
    """
    To train in a standard training loop make sure to modify the train function so you pass in the inputs and the labels
    i.e. output = model(images, labels)
    """

    def __init__(self, model, margin=0.5, scaler=64, embedding_size=NotImplemented, num_classes=NotImplemented):
        super(ArcFaceModel, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes

        # small number to avoid invalid arcCos values
        self.eps = 1e-7

        # hyperparameters
        self.margin = margin
        self.scaler = scaler

        # load classification model
        self.model = model

        # Initializing the arcface linear layer with the weights of the classifier from the trained CNN
        self.AFL_linear = torch.nn.Linear(embedding_size, num_classes,
                                          bias=False)  # Why set bias=False? Check out the paper.
        with torch.no_grad():
            self.AFL_linear.weight.copy_(self.model.fc.weight)

        # Initializing utility functions for normalization, arcCos, cos and onehot encoding
        self.normalizer = torch.nn.functional.normalize
        self.arcCos = torch.acos
        self.cos = torch.cos
        self.one_hot = torch.nn.functional.one_hot

    def forward(self, x, labels=None):
        # Get face embedding.
        embedding = self.model(x, return_feats=True)

        # Normalize face embedding
        embedding = self.normalizer(embedding, dim=1)

        # Normalize linear layer weights
        with torch.no_grad():
            self.AFL_linear.weight = torch.nn.Parameter(self.normalizer(self.AFL_linear.weight, dim=1))

        # Reshape
        embedding = embedding.view(-1, self.embedding_size)

        # Take dot product to get cos theta
        cosine = F.linear(embedding, self.AFL_linear.weight)

        # If labels are not provided, return the cosine values (or embeddings)
        # if labels is None:
        #     return cosine

        # Clamp the cosine values
        cosine = torch.clamp(cosine, min=-1.0 + self.eps, max=1.0 - self.eps)

        # Get theta by performing arccos(cos(theta))
        theta = self.arcCos(cosine)

        # Convert labels to one-hot
        # one_hot_labels = labels
        one_hot_labels = self.one_hot(labels, self.num_classes).float()

        # Create a mask with m at positions with label 1 and 0 at positions with label 0
        margin_mask = one_hot_labels * self.margin

        # Add margin m to theta
        theta_m = theta + margin_mask

        # Calculate the cosine value for theta with margin added and scale with self.scaler
        logits = self.scaler * self.cos(theta_m)

        return logits

class ArcFaceSEnet(torch.nn.Module):
    def __init__(self, arcface_model, num_classes):
        super(ArcFaceSEnet, self).__init__()
        self.base_model = arcface_model.model  # Extract the base model from ArcFace
        self.fc = torch.nn.Linear(arcface_model.embedding_size, num_classes)

    def forward(self, x, labels=None):
        x = self.base_model(x, return_feats=True)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

