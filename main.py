import math
import os
import pickle
import re

import librosa
import librosa.display
import numpy as np
import pandas as pd
import shutup
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


file_path = 'C:/Users/avata/Desktop/GP/IEMOCAP_full_release/Session1/dialog/EmoEvaluation/'
file_path2 = 'C:/Users/avata/Desktop/GP/IEMOCAP_full_release/Session1/dialog/wav/'


labels_df = pd.DataFrame(data=Read_labels(file_path))
audio_vector = Read_audio(file_path2, labels_df)


def extract_audio_features(Audio_vectors,labels_df, emotion_dict):
    df = pd.DataFrame(columns=['wav_file','mfcc','label'])
    for index, row in labels_df.iterrows():
        wav_file_name = row['wav_file']
        label = emotion_dict[row['emotion']]
        mfcc = np.mean(librosa.feature.mfcc(y=Audio_vectors[wav_file_name], sr=sr, n_mfcc=1).T, axis=0)
        # append(pd.DataFrame(feature_list, index=columns).transpose(), ignore_index=True)
        df = df.append(pd.DataFrame([wav_file_name, mfcc, label], index=['wav_file', 'mfcc', 'label']).transpose(), ignore_index=True)
    with open('C:/Users/avata/Desktop/SPEECH MODEL/data/audios1.pkl', 'wb') as f:
        pickle.dump(df, f)

    # df['mfcc'] = df['mfcc'].infer_objects(convert_numeric=True)
    return df


# def labeling():


# read the pickle and the csv


# labels_df = pd.read_csv('C:/Users/avata/Desktop/SPEECH MODEL/data/df_iemocap.csv')
# audio_vectors = pickle.load(open('C:/Users/avata/Desktop/SPEECH MODEL/data/audio_vectors1.pkl', 'rb'))
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
