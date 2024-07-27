import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
from keras._tf_keras.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Global variables for recording
audio_data = None
current_time = 0
recording = None
models = {}  # Define models as a dictionary
class_counts = {}  # Define class_counts as a dictionary

def preprocess_audio(audio_data, sr=44100, n_mfcc=13, desired_length=1222):
    # Extract features (example using MFCC)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
    mfccs_flat = mfccs.ravel()  # Flatten MFCCs if necessary
    
    # Ensure exactly desired_length features
    if len(mfccs_flat) < desired_length:
        # Pad with zeros if fewer features
        mfccs_flat = np.pad(mfccs_flat, (0, desired_length - len(mfccs_flat)), mode='constant')
    elif len(mfccs_flat) > desired_length:
        # Truncate if more features
        mfccs_flat = mfccs_flat[:desired_length]
    
    return mfccs_flat

def record_audio(file_name, duration=20):
    global audio_data, recording

    sample_rate = 44100
    channels = 2

    print(f"Recording audio for {duration} seconds... Press Enter to stop.")

    # Initialize audio_data array
    audio_data = np.empty((duration * sample_rate, channels), dtype=np.float32)

    try:
        recording = sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback)
        recording.start()

        print(f"Recording started. Press Enter to stop.")
        input()  # Wait for Enter to stop recording

    except KeyboardInterrupt:
        print('\nRecording stopped.')

    finally:
        if recording:
            recording.stop()
            recording.close()

            # Trim audio_data to the actual recorded length
            audio_data = audio_data[:current_time]

            # Save recorded audio to file
            sf.write(file_name, audio_data, sample_rate)

            print(f"Audio saved as '{file_name}'")

def callback(indata, frames, time, status):
    global audio_data, current_time

    if status:
        print(status)

    audio_data[current_time:current_time + frames] = indata
    current_time += frames

def detect_pauses(audio_data, sample_rate, threshold=0.02, min_silence=0.1):
    pauses = []
    is_silence = True
    silence_start = 0

    # Normalize audio data for better thresholding
    audio_data_norm = audio_data / np.max(np.abs(audio_data))

    for i, amp in enumerate(audio_data_norm):
        if abs(amp) > threshold:
            if is_silence:
                silence_end = i / sample_rate
                if silence_end - silence_start >= min_silence:
                    pauses.append((silence_start, silence_end))
                is_silence = False
        else:
            if not is_silence:
                silence_start = i / sample_rate
                is_silence = True

    # Check for silence at the end of audio
    if is_silence:
        silence_end = len(audio_data_norm) / sample_rate
        if silence_end - silence_start >= min_silence:
            pauses.append((silence_start, silence_end))

    return pauses

if __name__ == "__main__":
    # Adjust the file name as needed
    file_name = "recorded_audio.wav"

    # Record audio and save to file
    record_audio(file_name, duration=20)  # Record for up to 20 seconds

    # Load models (example loading models and scalers)
    models = {
        'SoundRep': load_model('sound.h5'),
        'WordRep': load_model('word.h5'),
        'Prolongation': load_model('prolongation.h5'),
        'Interjection': load_model('interjection.h5')
    }

    # Load scalers for each model (example loading scalers)
    scalers = {
        'SoundRep': joblib.load('scalersound.pkl'),
        'WordRep': joblib.load('scalerword.pkl'),
        'Prolongation': joblib.load('scalerpro.pkl'),
        'Interjection': joblib.load('scalerinter.pkl')
    }


    # Example: Preprocess the recorded audio
    audio_data, sr = librosa.load(file_name, sr=44100)

    # Detect pauses in the recorded audio
    pauses = detect_pauses(audio_data, sr, threshold=0.02, min_silence=0.1)

    # Print detected pauses
    print(f"Number of Pauses Detected: {len(pauses)}")
    for idx, (start, end) in enumerate(pauses):
        print(f"Pause {idx + 1}: Start at {start:.2f} seconds, End at {end:.2f} seconds")

    # Segment the audio into 4-second chunks and process each segment
    segment_duration = 4  # seconds
    segment_length = sr * segment_duration
    num_segments = int(len(audio_data) / segment_length)

    class_counts = {}
    for model_name, model in models.items():
        scaler = scalers[model_name]
        total_class_1_count = 0

        for i in range(num_segments):
            start_idx = i * segment_length
            end_idx = start_idx + segment_length
            audio_segment = audio_data[start_idx:end_idx]

            # Preprocess audio segment
            mfccs_flat = preprocess_audio(audio_segment)

            if len(mfccs_flat) != 1222:
                print(f"Warning: Audio segment {i+1} features length {len(mfccs_flat)} is not 1222, skipping prediction.")
                continue

            # Reshape and normalize using the corresponding scaler
            X_input = mfccs_flat.reshape(1, -1)  # Reshape for scaler
            X_input_normalized = scaler.transform(X_input)

            # Predict probabilities
            y_pred_prob = model.predict(X_input_normalized)

            # Example assuming class 1 is index 1 in the output probabilities
            class_1_count = np.sum(y_pred_prob[:, 1])  # Count instances of class 1
            total_class_1_count += class_1_count

        class_counts[model_name] = total_class_1_count / num_segments  # Average count per segment

    # Print results
    for model_name, count in class_counts.items():
        print(f"Average Class 1 count for {model_name}: {count:.2f}")

