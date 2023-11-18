import librosa
import numpy as np
audio_file_path = "ee.wav"
audio, sr = librosa.load(audio_file_path)
hop_length = 512  # Adjust hop length as needed
n_fft = 2048  # Adjust n_fft as needed

S = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
mag = np.abs(S)

# Define a threshold for identifying peak spectrogram region
threshold = 0.5  # Adjust threshold value as needed

peak_mag = mag > threshold

# Find the starting and ending indices of the peak spectrogram region
start_index = np.where(peak_mag.any(axis=0))[0][0]
end_index = np.where(peak_mag.any(axis=0))[0][-1] + 1

# Calculate the time duration of each frame based on hop length and sample rate
time_step = hop_length / sr

# Calculate the start and end timestamps for the audio segment
start_time = start_index * time_step
end_time = end_index * time_step

# Extract the corresponding audio segment
segment = audio[int(start_time * sr):int(end_time * sr)]

file_path = r"Output.wav"  # Use 'r' prefix for raw string to avoid escaping issues
librosa.audio.save_file(file_path, segment, sr)

