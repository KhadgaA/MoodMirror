
import os
from json_tricks import load

import numpy as np

import librosa
from pydub import AudioSegment, effects
import noisereduce as nr

import tensorflow as tf
import keras
from keras.models import model_from_json
from keras.models import load_model

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

saved_model_path = r'./model8723.json'
saved_weights_path = r'./model8723_weights.h5'

#Reading the model from JSON file
with open(saved_model_path, 'r') as json_file:
    json_savedModel = json_file.read()
    
# Loading the model architecture, weights
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights(saved_weights_path)

# Compiling the model with similar parameters as the original model.
model.compile(loss='categorical_crossentropy', 
                optimizer='RMSProp', 
                metrics=['categorical_accuracy'])

print(model.summary())

def convert(y,sr):
    # convert from float to uint16
    y = np.array(y * (1<<15), dtype=np.int16)
    audio_segment = AudioSegment(
        y.tobytes(), 
        frame_rate=sr,
        sample_width=y.dtype.itemsize, 
        channels=1
    )
    return audio_segment

def preprocess(y,sr ):
    
    '''
    A process to an audio .wav file before execcuting a prediction.
      Arguments:
      - file_path - The system path to the audio file.
      - frame_length - Length of the frame over which to compute the speech features. default: 2048
      - hop_length - Number of samples to advance for each frame. default: 512

      Return:
        'X_3D' variable, containing a shape of: (batch, timesteps, feature) for a single file (batch = 1).
    '''
    total_length = 204288 
    frame_length = 2048
    hop_length = 512
    # Fetch sample rate.
    # _, sr = librosa.load(path = file_path, sr = None)
    # Load audio file
    rawsound = convert(y,sr)
    # y = y.astype(np.float32)
    # y /= np.max(np.abs(y))

    # rawsound = AudioSegment.from_mono_audiosegments(y)
    # Normalize to 5 dBFS 
    normalizedsound = effects.normalize(rawsound, headroom = 5.0) 
    # Transform the audio file to np.array of samples
    normal_x = np.array(normalizedsound.get_array_of_samples(), dtype = 'float32') 
  
    final_x = nr.reduce_noise(normal_x, sr=sr) #updated 03/03/22
       
    # Features extraction 
    f1 = librosa.feature.rms(y = final_x, frame_length=frame_length, hop_length=hop_length,center=True,pad_mode='reflect').T # Energy - Root Mean Square   
    f2 = librosa.feature.zero_crossing_rate(final_x , frame_length=frame_length, hop_length=hop_length, center=True).T # ZCR      
    f3 = librosa.feature.mfcc(y = final_x, sr=sr, n_mfcc=13, hop_length = hop_length).T # MFCC

    X = np.concatenate((f1, f2, f3), axis = 1)
    # Pad the array
    padding_rows = 448-len(X)
    X = np.vstack(( X, np.zeros((padding_rows, 15))))

    X_3D = np.expand_dims(X, axis=0)
    
    return X_3D

emotions = {
    0 : 'neutral',
    1 : 'calm',
    2 : 'happy',
    3 : 'sad',
    4 : 'angry',
    5 : 'fearful',
    6 : 'disgust',
    7 : 'suprised'   
}
emo_list = list(emotions.values())

def is_silent(data):
    # Returns 'True' if below the 'silent' threshold
    return max(data) < 100
import pyaudio
import wave
from array import array
import struct
import time

# Initialize variables
RATE = 24414
CHUNK = 512
RECORD_SECONDS = 7.1

CHANNELS = 1
WAVE_OUTPUT_FILE = "./output.wav"


def EmotionRecogniser(stream,new_chunk):
    # process only when stream gets to length 7.1 seconds, else donot update prediction yet
    sr, y = new_chunk
    
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    # SESSION START
    print("** session started")
    total_predictions = [] # A list for all predictions in the session.
    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    # if len(stream) < int(RATE*RECORD_SECONDS):
    #     return stream, 'neutral'

    x = preprocess(y=stream,sr =sr) # 'output.wav' file preprocessing.
    print('x shape:', x.shape)
    # Model's prediction => an 8 emotion probabilities array.
    predictions = model.predict(x, use_multiprocessing=True)
    pred_list = list(predictions)
    pred_np = np.squeeze(np.array(pred_list).tolist(), axis=0) # Get rid of 'array' & 'dtype' statments.
    total_predictions.append(pred_np)
    
    #dict of emotions with their respective probabilities
    emotions_prob = dict(zip(emo_list, pred_np))
    max_emo = np.argmax(predictions)
    print('max emotion:', emotions.get(max_emo,-1))

    stream = stream[len(y):] # Reset the stream for the next session.
    emotions_prob

    return stream , emotions_prob

        # Present emotion distribution for the whole session.
        # total_predictions_np =  np.mean(np.array(total_predictions).tolist(), axis=0)
        # fig = plt.figure(figsize = (10, 5))
        # plt.bar(emo_list, total_predictions_np, color = 'indigo')
        # plt.ylabel("Mean probabilty (%)")
        # plt.title("Session Summary")
        # plt.show()

        # print(f"Emotions analyzed for: {(toc - tic):0.4f} seconds")
        # return str(emotions.get(np.argmax(total_predictions_np),-1))

##################################################

import gradio as gr
from transformers import pipeline
import numpy as np

# transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

# def transcribe(stream, new_chunk):
#     sr, y = new_chunk
#     y = y.astype(np.float32)
#     y /= np.max(np.abs(y))

#     if stream is not None:
#         stream = np.concatenate([stream, y])
#     else:
#         stream = y
#     return stream, transcriber({"sampling_rate": sr, "raw": stream})["text"]


demo = gr.Interface(
    EmotionRecogniser,
    ["state",gr.Audio(sources=["microphone"], streaming=True,every=1.0)],
    ["state",'label'],
    live=True,
)

demo.launch()
