import torch
from torch import nn


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_lemmas, embedding_dim):
        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()
        self.lemma_embedding = nn.Embedding(num_lemmas + 1, embedding_dim)  # +1 to handle no lemma

    def forward(self, x, lemma_ids):
        lemma_embeds = self.lemma_embedding(lemma_ids)
        lemma_embeds = lemma_embeds.view(lemma_embeds.shape[0], -1)
        x = torch.cat((x, lemma_embeds), dim=1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
