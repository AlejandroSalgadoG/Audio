from pygame import mixer

from Wav.WavWriter import WavWriter


class WavReproducer:
    def __init__(self):
        mixer.init()

    def reproduce(self, time_data):
        mixer.music.load( WavWriter(time_data).get_file_obj() )
        mixer.music.play()