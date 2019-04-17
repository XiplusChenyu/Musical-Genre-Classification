import librosa
import torch
import numpy as np

from basic_model import M_model
from Paras import Para


class MusicDealer:
    def __init__(self, weight_path, model):
        self.model = model
        model.load_state_dict(torch.load(weight_path, map_location='cpu'))

    @staticmethod
    def audio_loader(audio_path):
        y, _ = librosa.load(audio_path, mono=True, sr=Para.sample_rate)
        y = librosa.feature.melspectrogram(y=y, sr=Para.sample_rate, n_mels=128).T
        y = librosa.power_to_db(y)
        y = y[:-1 * (y.shape[0] % 128)]
        chunk_num = int(y.shape[0] / 128)
        mel_chunks = np.split(y, chunk_num)
        return mel_chunks

    def get_genre(self, audio_path):
        print("Test on {0}".format(audio_path.split('/')[-1]))
        tag_list = dict()
        idx_list = [x for x in range(10)]
        for idx in idx_list:
            tag_list[idx] = 0

        for i, data in enumerate(self.audio_loader(audio_path)):
            data = torch.FloatTensor(data).view(1, 1, 128, 128)  # resize to fit it in model
            predict = self.model(data)
            score, tag = predict.max(1)
            tag_list[int(tag)] = tag_list.get(int(tag)) + float(score)

        _sum = sum([tag_list.get(key) for key in tag_list])

        idx_list.sort(key=lambda x: -1 * tag_list.get(x))
        for i in idx_list:
            current_genre = Para.dictionary.get(i)
            current_score = tag_list.get(i) / _sum * 100
            if current_score == 0:
                break
            print('Genre {0}: {1}%'.format(current_genre, round(current_score, 2)))


if __name__ == '__main__':
    WEIGHT_PATH = "../model/best_model_1.pt"
    TEST_FILE = "../sample_music/Kendrick Lamar - HUMBLE..mp3"

    dealer = MusicDealer(WEIGHT_PATH, M_model)
    dealer.get_genre(TEST_FILE)
