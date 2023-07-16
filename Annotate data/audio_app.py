import streamlit as st
from pydub import AudioSegment
from io import BytesIO
import os
import pandas as pd
import numpy as np
df = pd.read_csv('data.csv')

def save_transcription(transcription, audio_file):
    df.loc[df['file'] == audio_file, 'edited'] = transcription
    df.to_csv('data.csv', index=False)


def main():
    audio_paths = ["../trimmed_audios/"+f for f in df['file'].tolist()]

    transcriptions = []
    for i in range(len(audio_paths)):
        if pd.isnull(df.at[i, "edited"]):
            transcriptions.append(df.iloc[i]["text"])
        else:
            transcriptions.append(df.iloc[i]["edited"])
    st.title("Moroccan dialect transcription")
    # st.write("Navigate through audio files and view their transcriptions")

    num_pairs = len(audio_paths)

    current_index = st.sidebar.number_input("Select an audio file (Index)", min_value=0, max_value=num_pairs-1, value=0, step=1)

    audio_file, transcription = audio_paths[current_index], transcriptions[current_index]

    audio = AudioSegment.from_file(audio_file)
    st.write(f"Audio: {audio_file}")

    audio.export("temp.mp3", format="mp3")
    st.audio("temp.mp3", format='audio/mp3')

    st.write("Transcription:")
    transcription_editable = st.text_area("Edit transcription", transcription)

    if st.button("Save"):
        # Save the edited transcription
        # Here, you can define your own logic to save the transcription
        save_transcription(transcription_editable, audio_file.replace("../trimmed_audios/", ""))



    if st.button("Previous") and current_index > 0:
        current_index -= 1

    if st.button("Next") and current_index < num_pairs-1:
        current_index += 1

    st.sidebar.write(f"Current audio index: {current_index}")


if __name__ == '__main__':
    main()