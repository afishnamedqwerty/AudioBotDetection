import os
import torch
import torchaudio
import tensorflow as tf
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.io.wavfile import write
from tqdm import tqdm

'''Tuning Sigma in WaveGlow Inference

The sigma parameter in the WaveGlow model inference call controls the variance 
of the Gaussian distribution used to model the audio data. Adjusting sigma can 
affect the audio quality and the characteristics of the synthesized speech:

    A lower sigma value can lead to clearer and more precise audio outputs but 
    may occasionally produce audio that sounds overly processed or unnatural.
    A higher sigma value may generate more natural-sounding audio with a 
    potential increase in background noise or less clarity.

The default value provided in the NVIDIA examples is often 0.666. This is a balanced 
choice for many applications, but depending on your specific requirements, you might 
want to experiment with slight adjustments. For high-quality, clear speech, you might 
start with values slightly lower, testing in small decrements (e.g., 0.65, 0.60) and 
listen to the output to find the best balance between clarity and naturalness.'''

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_csv(file_path):
    df = pd.read_csv(file_path, sep='|', header=None, names=['ID', 'Text', 'Normalized_Text'], na_filter=False)
    return df['Text'].tolist(), df['ID'].tolist()

# Load Tacotron2 and WaveGlow models
#tacotron2 = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2', map_location=device).eval()
#waveglow = torch.hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow', map_location=device).eval()

# Using the Tacotron2 and WaveRNN bundled pipeline for phoneme-based processing
bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
processor = bundle.get_text_processor()
tacotron2 = bundle.get_tacotron2().to(device).eval()
vocoder = bundle.get_vocoder().to(device).eval()

def synthesize_text(text):
    # Disable autograd
    with torch.no_grad():
        
        sequence = text_to_speech(text)
        #sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.from_numpy(sequence).to(device, dtype=torch.int64)
        
        # Synthesize mel spectrogram
        mel_outputs, mel_outputs_postnet, _, _ = tacotron2.inference(sequence)
        
        # Synthesize audio from mel spectrogram using WaveGlow
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
        
        audio_numpy = audio[0].data.cpu().numpy()
        return audio_numpy, 22050  # Tacotron 2 default sampling rate

def save_audio(audio, sample_rate, file_name):
    sf.write(file_name, audio, sample_rate)


def synthesize_dataset(csv_path, output_dir):
    texts, ids = parse_csv(csv_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Wrap the zipping of texts and ids with tqdm for a progress bar
    for text, id in tqdm(zip(texts, ids), total=len(texts), desc="Synthesizing dataset"):
        try:
            audio_numpy, sample_rate = text_to_speech(text)
            file_name = os.path.join(output_dir, f"{id}.wav")
            save_audio(audio_numpy, sample_rate, file_name)
            # Removed the print statement to keep the output clean with tqdm
        except Exception as e:
            print(f"Error processing {id}: {e}")

def text_to_speech(text):
    # Processing text to phoneme-based tokens
    with torch.inference_mode():
        processed, lengths = processor(text)
        processed = processed.to(device) 
        lengths = lengths.to(device)
        #lengths = lengths.to(device)

        # Generating spectrogram using Tacotron2
        spec, spec_lengths, _ = tacotron2.infer(processed, lengths)

        # Generating waveform using the vocoder
        waveforms, _ = vocoder(spec, spec_lengths)
    
    return waveforms.squeeze().cpu(), vocoder.sample_rate

'''def synthesize_dataset(csv_path, output_dir):
    texts, ids = parse_csv(csv_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for text, id in zip(texts, ids):
        try:
            audio, sample_rate = synthesize_text(text)
            file_name = os.path.join(output_dir, f"{id}.wav")
            save_audio(audio, sample_rate, file_name)
            print(f"Saved {file_name}")
        except Exception as e:
            print(f"Error processing {id}: {e}")'''

if __name__ == "__main__":
    csv_path = "./ljs_dataset/LJSpeech-1.1/metadata.csv"
    extract_to = "./synthetic_ljs"

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    print(torch.cuda.is_available())  # Should return True
    print(torch.version.cuda)  # Check the CUDA version PyTorch was built with
    synthesize_dataset(csv_path, extract_to)
