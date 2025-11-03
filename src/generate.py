import torch
from model import MusicTransformer
import miditok
import json
import os

config = json.load(open("config.json"))

model = MusicTransformer(
    vocab_size=config["vocab_size"],
    embed_dim=config["embed_dim"],
    n_heads=config["n_heads"],
    n_layers=config["n_layers"]
)
model.load_state_dict(torch.load(f"{config['model_dir']}/music_transformer.pt"))
model.eval()

def generate_music(model, seed_token, max_length=512):
    generated = torch.tensor([seed_token]).unsqueeze(0)
    for _ in range(max_length):
        out = model(generated)
        next_token = torch.argmax(out[0, -1], dim=-1).unsqueeze(0).unsqueeze(0)
        generated = torch.cat([generated, next_token], dim=1)
    return generated.squeeze().tolist()

seed = 10  # 랜덤 시작
tokens = generate_music(model, seed)

tokenizer = miditok.MIDILike()
decoded = tokenizer.tokens_to_midi([int(i) for i in tokens])
os.makedirs(config["output_dir"], exist_ok=True)
decoded.dump(f"{config['output_dir']}/generated_song.mid")
print("🎵 Generated song saved!")
