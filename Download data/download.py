from pytube import YouTube
import tqdm


def youtube_download():
    f = open("links.txt", "r")
    s = 0
    results = f.readlines()
    for link in tqdm.tqdm(results):
        try:
            vid = YouTube(link)
            print(link)
            l = vid.length
            if True:
                vid = vid.streams.get_audio_only()
                vid.download(output_path= "audio_podcast")

                s += l
                print(str(l) + "-------" + str(s))
        except:
            print("video skipped")
            pass    
    print("================================")
    print("Length : "+ str(s))
    print("================================")
    # open('links.txt', 'w').close()

youtube_download()