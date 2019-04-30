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

### Test
Use music_dealer.py to predict the genre components of full song

## Result
- Model: Pure CNN Model
- Precise: 
  - Accuracy: 86.17%
- Sample prediction:
```
Test on Kendrick Lamar - HUMBLE..mp3
Genre hiphop: 58.09%
Genre pop: 31.78%
Genre disco: 4.6%
Genre reggae: 2.94%
Genre blues: 2.6%
***************************************************************************************************
Test on Jackson Wang- Papillon.mp3
Genre pop: 60.93%
Genre reggae: 19.63%
Genre hiphop: 16.87%
Genre blues: 1.29%
Genre disco: 1.28%
***************************************************************************************************
Test on Adele - Hello.mp3
Genre pop: 61.9%
Genre blues: 19.15%
Genre country: 11.72%
Genre reggae: 3.87%
Genre hiphop: 2.67%
Genre metal: 0.7%
```
