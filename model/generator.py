# model/generator.py
import torch
from miditok import REMI
from miditoolkit import MidiFile
from model.music_transformer import MusicTransformer
import random, os

# ==============================
# 🎵 Tokenizer 초기화
# ==============================
tokenizer = REMI()

# vocab 설정 (BPE가 없을 경우 기본 vocab 사용)
if hasattr(tokenizer, "vocab_bpe") and tokenizer.vocab_bpe is not None:
    vocab_size = len(tokenizer.vocab_bpe)
else:
    vocab_size = len(tokenizer.vocab)

# ==============================
# 🎶 모델 초기화
# ==============================
model = MusicTransformer(vocab_size=vocab_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ==============================
# 🎼 음악 생성 함수
# ==============================
def generate_music(genre="pop", mood="happy", duration=10):
    model.eval()

    # 1️⃣ 시작 토큰 설정
    start_token = torch.tensor([[0]], device=device)  # 단순한 시작 토큰
    seq = start_token
    generated = [start_token.item()]

    # 2️⃣ 토큰 생성 (샘플 예시: 50개)
    for _ in range(50):
        with torch.no_grad():
            logits = model(seq)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
        generated.append(next_token.item())
        seq = torch.cat([seq, next_token.unsqueeze(0)], dim=1)

    # 3️⃣ MIDI 변환
    vocab_len = vocab_size
    decoded_tokens = [int(tok) % vocab_len for tok in generated]

    try:
        midi = tokenizer.tokens_to_midi(decoded_tokens)
    except Exception as e:
        print(f"❌ MIDI 변환 실패: {e}")
        midi = MidiFile()

    # 4️⃣ 파일 저장
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", f"{genre}_{mood}.mid")
    midi.dump(out_path)

    print(f"✅ 생성 완료: {out_path}")
    return out_path
