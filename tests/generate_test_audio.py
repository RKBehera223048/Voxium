import wave
import struct
import math
import os

def generate_test_audio(filename="tests/test_audio.wav"):
    sample_rate = 16000
    duration = 1.0  # seconds
    frequency = 440.0  # Hz

    with wave.open(filename, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        for i in range(int(sample_rate * duration)):
            value = int(32767.0 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)

if __name__ == "__main__":
    generate_test_audio()
