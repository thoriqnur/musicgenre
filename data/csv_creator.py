import librosa
import numpy as np
import os
import csv
import warnings
import sys
warnings.filterwarnings('ignore')

db_path = sys.argv[0]

# Creating Header file
header = 'filename bpm chroma_stft rms spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'

# for i in range(1, 21):
#     header += 'mfcc' + str(i)

header += ' mfcc1 mfcc2 mfcc3 mfcc4 mfcc5 mfcc6 mfcc7 mfcc8 mfcc9 mfcc10 mfcc11 mfcc12 mfcc13 mfcc14 mfcc15 mfcc16 mfcc17 mfcc18 mfcc19 mfcc20'

header += ' label'
header = header.split()


# export csv file
file = open('./data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

print("analyse")
sys.stdout.flush()

count = 1;

for g in genres:
    for filename in os.listdir(f'{db_path}/{g}'):
        song_name = f'{db_path}/{g}/{filename}'
        y, sr = librosa.load(song_name, mono=True, duration=30)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        bpm, beats = librosa.beat.beat_track(y=y, sr=sr)
        to_append = f'{filename} {bpm} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'
        file = open('./data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())
        percents = round(100.0 * count / 1000, 1)
        count += 1
        print(f'{filename}     {percents}%')
        sys.stdout.flush()

print("finish")
sys.stdout.flush()
