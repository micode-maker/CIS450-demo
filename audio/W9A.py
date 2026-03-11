# W9A.py: Normalize, Trim Silence, 2X Speed Up
# LICENSE = BSD-3-Clause

import numpy as np
from scipy.io import wavfile

def normalize_audio(audio):
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio

def trim_silence(audio, threshold=0.1):
    mask = np.abs(audio) > threshold
    if not np.any(mask):
        return audio
    start_idx = np.where(mask)[0][0]
    end_idx = np.where(mask)[0][-1]
    return audio[start_idx:end_idx]

def change_speed(audio, rate=2.0):
    indices = np.arange(0, len(audio), rate).astype(int)
    return audio[indices]

def main():
    try:
        samplerate, data = wavfile.read("output.wav")
        
        audio = data.astype(np.float32) / np.iinfo(data.dtype).max if data.dtype != np.float32 else data

        normalized = normalize_audio(audio)

        trimmed = trim_silence(normalized, 0.1)

        sped_up = change_speed(trimmed, 2.0)

        final_audio = (sped_up * 32767).astype(np.int16)
        wavfile.write("processed.wav", samplerate, final_audio)
        
        print("Saved processed.wav")

    except FileNotFoundError:
        print("Error: output.wav not found in current directory.")

if __name__ == "__main__":
    main()