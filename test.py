import os
import pretty_midi

def check_midi_files(base_path):
    for genre in os.listdir(base_path):
        genre_path = os.path.join(base_path, genre)
        print(f"\n🎵 Checking {genre} files...")
        
        for file in os.listdir(genre_path):
            if file.endswith(".mid") or file.endswith(".midi"):
                try:
                    midi = pretty_midi.PrettyMIDI(os.path.join(genre_path, file))
                    duration = midi.get_end_time()
                    if duration < 10:
                        print(f"⚠️ Skipping {file} (too short: {duration:.2f}s)")
                    else:
                        print(f"✅ {file} - {duration:.2f}s")
                except Exception as e:
                    print(f"❌ Error in {file}: {e}")

# 실행 예시
check_midi_files("data")