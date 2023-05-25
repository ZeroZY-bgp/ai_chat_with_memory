import pyttsx3
import librosa
import soundfile as sf
import sounddevice as sd
from gtts import gTTS


class AudioModule:

    def __init__(self, sound_library='local', rate=130):
        if sound_library not in ['local', 'gtts']:
            raise AttributeError("语音库选择参数出错！传入的参数为", sound_library, "可选参数：", ['local', 'gtts'])
        self.sound_library = sound_library
        if self.sound_library == 'local':
            self.engine = pyttsx3.init()
            # 获取当前系统支持的所有发音人
            voices = self.engine.getProperty('voices')
            # 设置发音人
            self.engine.setProperty('voice', voices[0].id)
            # 设置语速
            self.engine.setProperty('rate', rate)
        # 调音高
        self.step = 4

    def say(self, text):
        audio_name = 'basic.wav'
        if self.sound_library == 'local':
            self.engine.save_to_file(text, audio_name)
            self.engine.runAndWait()
        elif self.sound_library == 'gtts':
            tts = gTTS(text, lang='zh-cn')
            tts.save(audio_name)

        audio_processed, sr = self.voice_process(audio_name)
        # 播放声音
        sd.play(audio_processed, sr)
        sd.wait()
        # sf.write('output.wav', audio_processed, sr)

    def voice_process(self, audio_name):
        y, sr = librosa.load(audio_name)
        audio_p = librosa.effects.pitch_shift(y, sr=sr, n_steps=self.step)
        return audio_p, sr
