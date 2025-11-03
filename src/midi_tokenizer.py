import os
import miditok
from tqdm import tqdm
import json

def tokenize_midi_files(data_dir, token_dir):
    os.makedirs(token_dir, exist_ok=True)
    tokenizer = miditok.MIDILike()

    for genre in os.listdir(data_dir):
        genre_path = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_path):
            continue

        for file in tqdm(os.listdir(genre_path), desc=f"Tokenizing {genre}"):
            if not file.endswith(".mid"):
                continue

            midi_path = os.path.join(genre_path, file)
            try:
                tokens = tokenizer(file_path=midi_path)
                output_path = os.path.join(token_dir, f"{genre}_{file}.json")
                tokens.save(output_path)
            except Exception as e:
                print(f"❌ Error with {file}: {e}")

if __name__ == "__main__":
    from pathlib import Path
    import json

    config = json.load(open("config.json"))
    tokenize_midi_files(config["data_dir"], config["token_dir"])
