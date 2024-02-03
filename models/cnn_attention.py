import torch.nn as nn
import torch.nn.functional as F
import torch
from cnn_residual import CustomResNet
from torchsummary import summary

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Adjusted the einsum equation to match the tensor dimensions
        energy = torch.einsum("nd,nd->n", [queries, keys])
        attention = F.softmax(energy / (self.embed_size ** (1 / 2)), dim=0)

        out = torch.einsum("n,nd->nd", [attention, values])
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# Your ResNet architecture remains unchanged

class HybridResNetTransformer(nn.Module):
    def __init__(self, num_classes=7001, embed_size=512, heads=8, dropout=0.5, forward_expansion=4):
        super(HybridResNetTransformer, self).__init__()
        self.conv_layers = CustomResNet()  # Your existing ResNet architecture
        self.embedding = nn.Linear(7001, embed_size)

        # Transformer blocks
        self.transformer_block1 = TransformerBlock(embed_size, heads, dropout, forward_expansion)
        self.transformer_block2 = TransformerBlock(embed_size, heads, dropout, forward_expansion)

        # Final classification layer
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.embedding(x)  # Convert to desired embed_size
        print("Before transformer:", x.shape)
        x = self.transformer_block1(x, x, x)
        print("After transformer block 1:", x.shape)
        x = self.transformer_block2(x, x, x)
        x = self.fc(x)
        return x


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CustomResNet().to(DEVICE)
model = HybridResNetTransformer(num_classes=7001).to(DEVICE)
# model.initialize_weights()
print(summary(model, (3, 224, 224)))