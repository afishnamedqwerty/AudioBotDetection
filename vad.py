import webrtcvad
import librosa

def vad_audio(y, sr, frame_duration=10):
    """Voice activity detection."""
    vad = webrtcvad.Vad()
    vad.set_mode(3)
    y_mono = librosa.to_mono(y)
    y_16bit = librosa.resample(y_mono, orig_sr=sr, target_sr=16000)
    frame_length = int(16000 * frame_duration / 1000)
    frames = [y_16bit[i:i+frame_length] for i in range(0, len(y_16bit), frame_length)]
    is_speech = [vad.is_speech(frame.tobytes(), 16000) for frame in frames]
    return is_speech
