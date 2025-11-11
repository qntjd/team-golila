import os
import pandas as pd
import pretty_midi
from tqdm import tqdm

DATA_DIR = "data"  # MIDI 파일이 들어 있는 상위 폴더
OUTPUT_CSV = "midi_metadata.csv"

def extract_midi_info(file_path):
    """MIDI 파일에서 템포, 길이, 악기 개수 추출"""
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)

        # 템포
        tempi = midi_data.get_tempo_changes()[1]
        avg_tempo = round(sum(tempi) / len(tempi), 2) if len(tempi) > 0 else 0

        # 길이 (초 단위)
        duration = round(midi_data.get_end_time(), 2)

        # 악기 개수
        num_instruments = len(midi_data.instruments)

        return avg_tempo, duration, num_instruments
    except Exception as e:
        print(f"❌ {os.path.basename(file_path)} 분석 오류: {e}")
        return None, None, None

def analyze_all_midi(data_dir):
    records = []

    for genre in os.listdir(data_dir):
        genre_path = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_path):
            continue

        print(f"\n🎶 {genre.upper()} 장르 분석 중...")

        for file in tqdm(os.listdir(genre_path)):
            if not file.endswith(".mid") and not file.endswith(".midi"):
                continue

            file_path = os.path.join(genre_path, file)
            tempo, duration, instruments = extract_midi_info(file_path)

            if tempo and duration:
                records.append({
                    "genre": genre,
                    "filename": file,
                    "tempo": tempo,
                    "duration_sec": duration,
                    "instruments": instruments
                })

    return pd.DataFrame(records)

def main():
    df = analyze_all_midi(DATA_DIR)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✅ 분석 완료! {len(df)}개 MIDI 정보가 {OUTPUT_CSV}에 저장되었습니다.")

if __name__ == "__main__":
    main()
