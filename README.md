# Musical Genre Classification
Three Models for Musical Genre Classification are discussed, one CNN model and two CRNN models. This project is based on Pytorch.

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

### Mel-Spectrogram Datasets
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
All models are over 80% on test set, see report for details

### Test on full songs
30 songs are used for test, Samples:
```
Test on Adele - Hello.mp3
Genre pop: 61.9%
Genre blues: 19.15%
Genre country: 11.72%
Genre reggae: 3.87%
Genre hiphop: 2.67%
Genre metal: 0.7%
```

# Thanks:
- https://www.mp3juices.cc/ for music download
