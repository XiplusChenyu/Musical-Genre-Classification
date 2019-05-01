import librosa
import torch
import numpy as np
from Paras import Para


class MusicDealer:
    def __init__(self, weight_path, model, size=128):
        self.model = model
        self.index = size
        self.remove = 15  # remove the first and last 15 s
        self.model.load_state_dict(torch.load(weight_path, map_location='cpu'))
        self.model = model.eval()

    def audio_loader(self, audio_path):
        y, _ = librosa.load(audio_path, mono=True, sr=Para.sample_rate)
        y = y[self.remove*Para.sample_rate: -1*self.remove*Para.sample_rate]
        y = librosa.feature.melspectrogram(y=y, sr=Para.sample_rate, n_mels=self.index).T
        y = librosa.power_to_db(y)
        if not y.shape[0] % self.index == 0:
            y = y[:-1 * (y.shape[0] % self.index)]
        chunk_num = int(y.shape[0] / self.index)
        mel_chunks = np.split(y, chunk_num)
        return mel_chunks

    def get_genre(self, audio_path):
        print("Test on {0}".format(audio_path.split('/')[-1]))
        tag_list = dict()
        idx_list = [x for x in range(10)]
        for idx in idx_list:
            tag_list[idx] = 0

        for i, data in enumerate(self.audio_loader(audio_path)):
            with torch.no_grad():
                data = torch.FloatTensor(data).view(1, 1, self.index, self.index)  # resize to fit it in model
                predict = self.model(data)
                score, tag = predict.max(1)
            tag_list[int(tag)] = tag_list.get(int(tag)) + float(score)

        _sum = sum([tag_list.get(key) for key in tag_list])

        idx_list.sort(key=lambda x: -1 * tag_list.get(x))
        tmp = dict()
        for i in idx_list:
            current_genre = Para.dictionary.get(i)
            current_score = tag_list.get(i) / _sum * 100
            tmp[i] = current_score
            if current_score == 0:
                break
            print('Genre {0}: {1}%'.format(current_genre, round(current_score, 2)))
        return idx_list[0], idx_list[1], idx_list[2], tmp


if __name__ == '__main__':
    from models import CnnModel, CrnnLongModel, CrnnModel

    path1 = 'mayday/如烟.mp3'
    path2 = 'mayday/转眼.mp3'

    WEIGHT_PATH = "../model/"
    dealer = MusicDealer(WEIGHT_PATH + "CrnnLongModel.pt", CrnnLongModel(), 256)

    dealer.get_genre(path1)
    dealer.get_genre(path2)
