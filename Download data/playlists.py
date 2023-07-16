from pytube import Playlist

def _getVideoLink():
    f = open("playlist.txt", "r")
    results = f.readlines()
    for playlist in results:
        print(playlist.title)
        f = open("links.txt", "a")
        playlist_urls = Playlist(playlist)
        for url in playlist_urls:
            f.write(url+"\n") 
    # open('playlist.txt', 'w').close()  

_getVideoLink()