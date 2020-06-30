# Deskripsi Project
- Topik : aplikasi prediksi genre musik
- Deskripsi : memprediksi genre musik country, classical, jazz, metal, pop, blues, rock, reggae, disco, dan hip-hop
- Data : ekstraksi manual fitur - fitur dari data audio pada http://marsyas.info/downloads/datasets.html (GTZAN)
- Feature extraction : menggunakan library librosa untuk mengekstrak fitur - fitur audio (cthnya MFCC)
- Feature selection : menggunakan PCA
- Algoritma yang digunakan :Naive Bayes Classifier

**Referensi** : http://cs229.stanford.edu/proj2016/poster/BurlinCremeLenain-MusicGenreClassification-poster.pdf

## Dataset

Fitur Musik yang digunakan adalah :
- MFCC (12 fitur)
- Chroma stft (12 fitur)
- Chroma cqt (12 fitur)
- Spectral Bandwidth (1 fitur)
- Spectral Rolloff (1 fitur)
- Spectral Contrast (7 fitur)

Cara Ekstraksi dapat dilihat pada https://librosa.github.io/librosa/feature.html, kemudian proses ekstraksi menggunakan [feature_extraction.py]

Sehingga terdapat 45 fitur audio/musik, masing - masing dihitung statistiknya (mean dan standard deviation) menghasilkan total fitur sebesar **90 fitur**

Prediksi hanya dilakukan pada genre musik pada GTZAN ----  ----

Dataset akhir dihasilkan 1000 baris (100 baris masing - masing genre) dan 91 kolom (1 kolom untuk label/target/genre dan 90 kolom untuk fitur yang sudah diekstraksi) 

## Implementasi Algoritma Klasifikasi

1. Naive Bayes Classifier


   
## Lain - Lain
Library luar yang digunakan adalah librosa, numpy, pandas dan sklearn
- librosa untuk ekstraksi fitur audio/musik
- numpy digunakan pada semua implementasi klasifikasi
- pandas digunakan untuk membuka atau membuat csv dan memisahkan kolom variabel/fitur dengan kolom output
- sklearn digunakan untuk memecah data ke data training dan testing, sklearn juga digunakan sebagai pembanding algoritma yang diimplementasikan sendiri 


- metrics, untuk melihat accuracy dan confusion matrix ---- [metrics.py](https://github.com/machine-learning-2018-2019-fasilkom-ui/Teknik-Mesin/blob/dev/learn/metrics.py) ----


