import matplotlib.pyplot as plt
import librosa.display

def plot_mfcc(audio_path, save_path):
    audio, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
def main():
    audio_name = "dxl1"
    audio_path = f"./deepfake_audio/{audio_name}.wav"
    plot_mfcc(audio_path, f"./{audio_name}")
    
if __name__ == "__main__":
    main()