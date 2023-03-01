import numpy as np
import pandas as pd
import os.path
import librosa
import IPython.display as ipd
from shutil import copyfile
import glob
import matplotlib.style as ms
import matplotlib.pyplot as plt
import shutup
from tqdm import tqdm
import pickle
import random
import time
import IPython.display
import librosa.display
import joblib
from joblib import Parallel, delayed
from PIL import Image
import multiprocessing as mp
import re
shutup.please()

info_line = re.compile(r'\[.+\]\n', re.IGNORECASE)


def read_df():
    start_times, end_times, wav_file_names, emotions, vals, acts, doms = [], [], [], [], [], [], []
    for sess in range(1, 6):
        emo_evaluation_dir = 'C:/users/avata/Desktop/GP/IEMOCAP_full_release/Session{}/dialog/EmoEvaluation/'.format(sess)
        evaluation_files = [l for l in os.listdir(emo_evaluation_dir) if 'Ses' in l]
        for file in evaluation_files:
            with open(emo_evaluation_dir + file) as f:
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

    df_iemocap.to_csv(
        'C:/Users/avata/Desktop/SPEECH MODEL/New data/df_iemocap_5.csv',
        index=False)

def build_audio_vectors():
    labels_df = pd.read_csv('C:/Users/avata/Desktop/SPEECH MODEL/New data/df_iemocap_5.csv')
    iemocap_dir = 'C:/users/avata/Desktop/GP/IEMOCAP_full_release/'
    sr = 44100
    audio_vectors = {}
    for sess in range(1,6):  # using one session due to memory constraint, can replace [5] with range(1, 6)
        wav_file_path = '{}Session{}/dialog/wav/'.format(iemocap_dir, sess)
        orig_wav_files = os.listdir(wav_file_path)
        for orig_wav_file in tqdm(orig_wav_files):
            try:
                orig_wav_vector, _sr = librosa.load(wav_file_path + orig_wav_file, sr=sr)
                orig_wav_file, file_format = orig_wav_file.split('.')
                for index, row in labels_df[labels_df['wav_file'].str.contains(orig_wav_file)].iterrows():
                    start_time, end_time, truncated_wav_file_name, emotion, val, act, dom = row['start_time'], row['end_time'], row['wav_file'], row['emotion'], row['val'], row['act'], row['dom']
                    audio_vectors[truncated_wav_file_name] = orig_wav_vector
            except:
                print(sess)
        with open('C:/Users/avata/Desktop/SPEECH MODEL/New data/audio_vectors_{}.pkl'.format(sess), 'wb') as f:
            pickle.dump(audio_vectors, f)


labels_df = pd.read_csv('C:/Users/avata/Desktop/SPEECH MODEL/New data/df_iemocap_5.csv')
audio_vectors_path = 'C:/Users/avata/Desktop/SPEECH MODEL/New data/audio_vectors_'

pickle_to_df = pd.DataFrame(columns=["filename", "audio_vector"])

for sess in range(1,2):
    audio_vectors = pickle.load(open('{}{}.pkl'.format(audio_vectors_path, sess), 'rb'))
    for index, row in tqdm(labels_df[labels_df['wav_file'].str.contains('Ses0{}'.format(sess))].iterrows()):
        wav_file_name = (row['wav_file'])
        y = audio_vectors[wav_file_name]
        # list_y = list(y)
        pickle_to_df = pickle_to_df.append({'filename': wav_file_name, 'audio_vector': list(y)}, ignore_index=True)

pickle_to_df.to_csv('C:/Users/avata/Desktop/SPEECH MODEL/New data/pickle_to_audiodf.csv')