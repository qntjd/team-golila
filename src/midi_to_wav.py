import os
import subprocess

def midi_to_wav(midi_path, soundfont="soundfont.sf2"):
    output = midi_path.replace(".mid", ".wav")
    cmd = f"fluidsynth -ni {soundfont} {midi_path} -F {output} -r 44100"
    os.system(cmd)
    print(f"✅ Converted {midi_path} → {output}")

if __name__ == "__main__":
    midi_to_wav("outputs/generated_song.mid")
