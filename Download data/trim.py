from pydub import AudioSegment
import os
import string
import random
import tqdm
import librosa
import soundfile as sf


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    # print("Random string of length", length, "is:", result_str)
    return result_str

i = 0

for file in tqdm.tqdm(os.listdir("audio_podcast")):
    path = f"audio_podcast/{file}"
    print(path)
# Load the audio file
    audio , sr = librosa.load(path)
    print(audio)
    start_sample = int(sr * 12)
    end_sample = start_sample + int(sr * 30)

# Trim the audio data

    while end_sample < len(audio) -  int(sr * 60):
        # cropped_audio.export("cropped_audio_file.wav", format="wav")
        trimmed_audio_data = audio[start_sample:end_sample]

        # wavfile.write('output_audio.wav', sample_rate, trimmed_audio_data)
        start_sample = end_sample + 1 * sr
        end_sample = end_sample + 31 * sr
        sf.write(f'New folder/{file}-{str(i+1)}.wav', trimmed_audio_data, sr)
        i += 1

