import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MIDIDataset
from model import MusicTransformer
import json, os

config = json.load(open("config.json"))

dataset = MIDIDataset(config["token_dir"])
loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

model = MusicTransformer(
    vocab_size=config["vocab_size"],
    embed_dim=config["embed_dim"],
    n_heads=config["n_heads"],
    n_layers=config["n_layers"]
)

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
loss_fn = nn.CrossEntropyLoss()

for epoch in range(config["epochs"]):
    for x, y in loader:
        out = model(x)
        loss = loss_fn(out.view(-1, out.size(-1)), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"✅ Epoch {epoch+1} | Loss: {loss.item():.4f}")

os.makedirs(config["model_dir"], exist_ok=True)
torch.save(model.state_dict(), f"{config['model_dir']}/music_transformer.pt")
