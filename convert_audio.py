import subprocess
import sys
import os

def convert_to_wav(input_file, output_file):
    """
    Convert an audio file (webm/ogg) to WAV format, 16kHz mono.
    Requires ffmpeg installed and in PATH.
    """
    if not os.path.isfile(input_file):
        print(f"Input file not found: {input_file}")
        return False
    
    command = [
        "ffmpeg",
        "-y",  # overwrite output file if exists
        "-i", input_file,
        "-ac", "1",        # mono audio
        "-ar", "16000",    # 16 kHz sample rate
        output_file
    ]
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Converted {input_file} -> {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e.stderr.decode()}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_audio.py input.webm output.wav")
    else:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        convert_to_wav(input_path, output_path)
