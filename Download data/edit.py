import random
import os
import string

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    # print("Random string of length", length, "is:", result_str)
    return result_str

for i in os.listdir("audio_podcast"):
    os.rename("audio_podcast/"+i, "audio_podcast/"+get_random_string(5)+".wav")