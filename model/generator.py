# model/generator.py
import os
import torch
import scipy.io.wavfile as wavfile
from transformers import MusicgenForConditionalGeneration, AutoProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("MusicGen device:", DEVICE)

MODEL_NAME = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)

def generate_music(genre="jazz", mood="happy", duration=10, prompt=None):
    if prompt is None:
        prompt = f"{genre} {mood} music"

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    base = f"{genre}_{mood}_{duration}"
    wav_path = os.path.join(out_dir, base + ".wav")

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    # MusicGen-small 기준: 1초 ≈ 40 tokens
    max_new_tokens = duration * 40

    audio_values = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens
    )

    # 모델 출력 → numpy
    audio = audio_values[0, 0].cpu().numpy()

    # WAV 저장
    wavfile.write(wav_path, 32000, audio)

    return wav_path
