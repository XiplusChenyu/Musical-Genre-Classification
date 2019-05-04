# Musical Genre Classification
Three Models for Musical Genre Classification are discussed, one CNN model and two CRNN models. This project is based on Pytorch. <a href="https://github.com/XiplusChenyu/Musical-Genre-Classification/blob/master/music_genre_classification.pdf">Project report here.</a>

## Get Start:
### Audio Files
- We use enhanced GTZAN Dataset, which contains 72 full songs and 1000 30s audio tracks.
- There are 10 genres, they are:
```
{0: 'pop',
 1: 'metal',
 2: 'disco',
 3: 'blues',
 4: 'reggae',
 5: 'classical',
 6: 'rock',
 7: 'hiphop',
 8: 'country',
 9: 'jazz'}
```

### Log Mel-Spectrogram Datasets
We offers three pre-processed datasets, you can also generate datasets using **Build Dataset Handmade.ipynb** or **Build Dataset.ipynb**. <a href='https://drive.google.com/file/d/1X3amA5n6RjYoY5QHdFfYOFfh9w3zbEU4/view?usp=sharing'>Download Here</a>
- Pure GTZAN Dataset (128^2 Chunks, 7000 in total)
- Mixed DatasetI (128^2 Chunks, 12370 in total)
- Mixed DatasetII (256^2 Chunks, 4533 in total)

### Training
- Define Parameters in Paras.py
- Use train.py for training
- Training Logs saved in log fold (loss/accuracy vs epoch on train set and validation set)

### Test
- Use music_dealer.py to predict the genre components of full song, see **genre_predictor.ipynb** and **music_dealer.py** for details
- Test result saved in log fold

## Result
### Test on frames
- Accuracy
<table>
  <tr>
    <th></th>
    <th>CNN Model</th>
    <th>CRNN-I Model</th>
    <th>CRNN-II Model</th>
  </tr>
  <tr>
    <td>Test Set</td>
    <td>88.05%</td>
    <td>85.08%</td>
    <td>88.45%</td>
  </tr>
  <tr>
    <td>Validation Set</td>
    <td>86.89%</td>
    <td>83.05%</td>
    <td>82.67%</td>
  </tr>
</table>

- Confusion Matrix

<img src='https://github.com/XiplusChenyu/Musical-Genre-Classification/blob/master/pictures/matrix.png' width=500>

### Test on full songs
30 songs are used for test, Samples:

<img src='https://github.com/XiplusChenyu/Musical-Genre-Classification/blob/master/pictures/sample%20prediction.png' width=250>

# Thanks:
- https://www.mp3juices.cc/ for music download
- https://github.com/cetinsamet/music-genre-classification
