import os
import pickle
import re
import shutup
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
from tqdm import tqdm
import librosa
import math
import random
import pandas as pd
import IPython.display
import librosa.display

ms.use('seaborn-muted')
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

shutup.please()
sr = 44100
emotion_dict = {'ang': 0,
                'hap': 1,
                'exc': 2,
                'sad': 3,
                'fru': 4,
                'fea': 5,
                'sur': 6,
                'neu': 7,
                'dis': 8,
                'xxx': 9,
                'oth': 9}


def Read_labels(datapath):
    info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)
    start_times, end_times, wav_file_names, emotions, vals, acts, doms = [], [], [], [], [], [], []

    evaluation_files = [l for l in os.listdir(datapath) if 'Ses' in l]
    for file in evaluation_files:
        with open(datapath + file) as f:
            content = f.read()
        info_lines = re.findall(info_line, content)
        for line in info_lines[1:]:  # the first line is a header
            start_end_time, wav_file_name, emotion, val_act_dom = line.strip().split('\t')
            start_time, end_time = start_end_time[1:-1].split('-')
            val, act, dom = val_act_dom[1:-1].split(',')
            val, act, dom = float(val), float(act), float(dom)
            start_time, end_time = float(start_time), float(end_time)
            start_times.append(start_time)
            end_times.append(end_time)
            wav_file_names.append(wav_file_name)
            emotions.append(emotion)
            vals.append(val)
            acts.append(act)
            doms.append(dom)

    df_iemocap = pd.DataFrame(columns=['start_time', 'end_time', 'wav_file', 'emotion', 'val', 'act', 'dom'])

    df_iemocap['start_time'] = start_times
    df_iemocap['end_time'] = end_times
    df_iemocap['wav_file'] = wav_file_names
    df_iemocap['emotion'] = emotions
    df_iemocap['val'] = vals
    df_iemocap['act'] = acts
    df_iemocap['dom'] = doms
    df_iemocap.to_csv('C:/Users/avata/Desktop/SPEECH MODEL/data/df_iemocap.csv', index=False)

    return df_iemocap


def Read_audio(datapath, labels_df):
    sr = 44100
    audio_vectors = {}
    wav_file_path = datapath
    orig_wav_files = os.listdir(wav_file_path)
    for orig_wav_file in tqdm(orig_wav_files):
        try:
            orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=sr)
            orig_wav_file, file_format = orig_wav_file.split('.')
            for index, row in labels_df[labels_df['wav_file'].str.contains(orig_wav_file)].iterrows():
                start_time, end_time, truncated_wav_file_name, emotion, val, act, dom = row['start_time'], row[
                    'end_time'], row['wav_file'], row['emotion'], row['val'], row['act'], row['dom']
                start_frame = math.floor(start_time * sr)
                end_frame = math.floor(end_time * sr)
                truncated_wav_vector = orig_wav_vector[start_frame:end_frame + 1]
                audio_vectors[truncated_wav_file_name] = truncated_wav_vector
        except:
            print('ai haga')
    with open('C:/Users/avata/Desktop/SPEECH MODEL/data/audio_vectors1.pkl', 'wb') as f:
        pickle.dump(audio_vectors, f)
    return audio_vectors


# making the labels and the audio vector


def audio_vector_and_labels():
    file_path = 'C:/Users/avata/Desktop/GP/IEMOCAP_full_release/Session1/dialog/EmoEvaluation/'
    file_path2 = 'C:/Users/avata/Desktop/GP/IEMOCAP_full_release/Session1/dialog/wav/'
    labels_df2 = pd.DataFrame(data=Read_labels(file_path))
    audio_vector = Read_audio(file_path2, labels_df2)


# End


def extract_audio_features2(Audio_vectors, labels_df, emotion_dict):
    df = pd.DataFrame(columns=['wav_file', 'mfcc', 'label'])
    for index, row in labels_df.iterrows():
        wav_file_name = row['wav_file']
        label = emotion_dict[row['emotion']]
        mfcc = np.mean(librosa.feature.mfcc(y=Audio_vectors[wav_file_name], sr=sr, n_mfcc=1).T, axis=0)
        # append(pd.DataFrame(feature_list, index=columns).transpose(), ignore_index=True)
        df = df.append(pd.DataFrame([wav_file_name, mfcc, label], index=['wav_file', 'mfcc', 'label']).transpose(),
                       ignore_index=True)
    with open('C:/Users/avata/Desktop/SPEECH MODEL/data/audios1.pkl', 'wb') as f:
        pickle.dump(df, f)

    # df['mfcc'] = df['mfcc'].infer_objects(convert_numeric=True)
    return df


# read the pickle and the csv


# labels_df = pd.read_csv('C:/Users/avata/Desktop/SPEECH MODEL/data/df_iemocap.csv')
# audio_vectors = pickle.load(open('C:/Users/avata/Desktop/SPEECH MODEL/data/audio_vectors1.pkl', 'rb'))
#
# random_file_name = list(audio_vectors.keys())[random.choice(range(len(audio_vectors.keys())))]
# y = audio_vectors[random_file_name]
# sr = 44100


def show():
    labels_df = pd.read_csv('C:/Users/avata/Desktop/SPEECH MODEL/data/df_iemocap.csv')
    audio_vectors = pickle.load(open('C:/Users/avata/Desktop/SPEECH MODEL/data/audio_vectors1.pkl', 'rb'))
    random_file_name = list(audio_vectors.keys())[random.choice(range(len(audio_vectors.keys())))]
    y = audio_vectors[random_file_name]
    # plt.figure(figsize=(15,2))
    # librosa.display.waveshow(y, sr=sr, max_points=1000, alpha=0.25, color='r')
    print('Signal mean = {:.5f}'.format(np.mean(abs(y))))
    print('Signal std dev = {:.5f}'.format(np.std(y)))
    rmse = librosa.feature.rmsz(y + 0.0001)[0]
    plt.figure(figsize=(15, 2))
    plt.plot(rmse)
    plt.ylabel('RMSE')
    print('RMSE mean = {:.5f}'.format(np.mean(rmse)))
    print('RMSE std dev = {:.5f}'.format(np.std(rmse)))
    plt.show()


def extract_audio_features(labels_path, audio_vector_path, emotion_dictionary, sessions,columns):
    labels_df = pd.read_csv(labels_path)
    audio_vector = pickle.load(open('C:/Users/avata/Desktop/SPEECH MODEL/data/audio_vectors1.pkl', 'rb'))
    for index, row in tqdm(labels_df[labels_df['wav_file'].str.contains('Ses01')].iterrows()):
        try:
            name = row['wav_file']
            label = emotion_dictionary[row['emotion']]
            y = audio_vector[name]

            #            start extracting features
            features_list = ['name', 'label']
            # signal mean and standard deviation
            signal_mean = np.mean(abs(y))
            signal_standard_deaviation = np.std(y)
            features_list.append(signal_mean)
            features_list.append(signal_standard_deaviation)
            # signal rmse mean and std
            rmse = librosa.feature.rms(y + 0.0001)[0]
            rmse_mean = np.mean(rmse)
            rmse_std = np.std(rmse)
            features_list.append(rmse_mean)
            features_list.append(rmse_std)

            # calculate silence from rmse and threshold
            silence = 0
            for energy in rmse:
                if energy <= 0.4 * rmse_mean:
                    silence += 1
            silence /= float(len(rmse))
            features_list.append(silence)
            harmonic = librosa.effects.hpss(y)[0]
            y_harmonic = np.mean(harmonic) * 1000
            features_list.append(y_harmonic)

            # auto correlation based on pitch detection
            center_clipper = 0.45 * signal_mean
            center_clipped = []
            for s in y:
                if s >= center_clipper:
                    center_clipped.append(s - center_clipper)
                elif s <= -center_clipper:
                    center_clipped.append(s + center_clipper)
                elif np.abs(s) < center_clipper:
                    center_clipped.append(0)
            auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
            features_list.append(1000 * np.max(auto_corrs) / len(auto_corrs))  # auto_corr_max (scaled by 1000)
            features_list.append(np.std(auto_corrs))  # auto_corr_std

            features = features.append(pd.DataFrame(features_list, index=columns).transpose(), ignore_index=True)
        except:
            print('ai haga')
# labeled_features_df = extract_audio_features(audio_vectors,labels_df,emotion_dict)
# labeled_features_df = pickle.load(open('C:/Users/avata/Desktop/SPEECH MODEL/data/audios1.pkl','rb'))
#
# for index,row in labeled_features_df.iterrows():
#     labeled_features_df['mfcc'][index] = np.array(labeled_features_df['mfcc'][index]).reshape(1,-1)
# x_train, x_test, y_train, y_test = train_test_split((labeled_features_df['mfcc']), labeled_features_df['label'],
#                                                     test_size=0.25)
# model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive',
#                       max_iter=500)
# # print(x_train[0].shape)
# x_train = np.array(x_train).reshape(-1,1)
# y_train = np.array(y_train).reshape(-1,1)
# print('shape is ',x_train.shape,' ',y_train.shape)
# model.fit(x_train,y_train.astype(int))
#
# y_pred = model.predict(x_test)
# accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
# print("Accuracy: {:.2f}%".format(accuracy * 100))

# break
