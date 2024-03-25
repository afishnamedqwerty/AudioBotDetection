from pyAudioAnalysis import audioSegmentation as aS

def diarize_audio(file_path):
    """Perform speaker diarization on an audio file."""
    segments = aS.speaker_diarization(file_path, n_speakers=0, mid_window=1.0, mid_step=0.1, short_window=0.05, lda_dim=35, plot_res=False, sm_segment=True)
    return segments
