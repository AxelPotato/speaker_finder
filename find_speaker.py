import os
import shutil
import argparse
from pathlib import Path
from pydub import AudioSegment
from resemblyzer import preprocess_wav, VoiceEncoder
from resemblyzer.audio import sampling_rate
import static_ffmpeg

# Ensure ffmpeg is set up
os.environ["PATH"] += os.pathsep + static_ffmpeg.get_ffmpeg_dir()


def load_wav(file_path):
    print(f"Loading audio file: {file_path}")
    # Convert MP3 to WAV if needed
    if file_path.endswith(".mp3"):
        print(f"Converting MP3 to WAV for file: {file_path}")
        audio = AudioSegment.from_mp3(file_path)
        file_path = file_path.replace(".mp3", ".wav")
        audio.export(file_path, format="wav")

    wav = preprocess_wav(file_path)
    return wav


def move_non_matching_files(reference_file, target_folder, destination_folder):
    # Load the reference audio file
    print(f"Loading reference audio file: {reference_file}")
    reference_wav = load_wav(reference_file)

    # Initialize the voice encoder
    print("Initializing the voice encoder")
    encoder = VoiceEncoder()

    # Embed the reference audio
    print("Embedding the reference audio")
    reference_embed = encoder.embed_utterance(reference_wav)

    # Ensure the destination folder exists
    Path(destination_folder).mkdir(parents=True, exist_ok=True)

    # Process each file in the target folder
    print(f"Processing files in the target folder: {target_folder}")
    for audio_file in os.listdir(target_folder):
        audio_file_path = os.path.join(target_folder, audio_file)

        # Check file extension and skip non-audio files
        if not (audio_file_path.endswith(".wav") or audio_file_path.endswith(".mp3")):
            print(f"Skipping non-audio file: {audio_file}")
            continue

        # Load and embed the target audio file
        try:
            print(f"Loading and embedding target audio file: {audio_file_path}")
            target_wav = load_wav(audio_file_path)
            target_embed = encoder.embed_utterance(target_wav)

            # Calculate similarity
            similarity = encoder.similarity(reference_embed, target_embed)
            print(f"Similarity for {audio_file}: {similarity:.4f}")

            # Move the file if similarity is below a certain threshold (you can adjust this threshold)
            if similarity < 0.75:  # Example threshold
                print(
                    f"Moving {audio_file} to {destination_folder} due to low similarity"
                )
                shutil.move(
                    audio_file_path, os.path.join(destination_folder, audio_file)
                )
        except Exception as e:
            print(f"Could not process {audio_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker verification script.")
    parser.add_argument(
        "--ref", type=str, required=True, help="Path to the reference audio file"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to the folder containing audio files to check",
    )
    parser.add_argument(
        "--move",
        type=str,
        required=True,
        help="Path to the folder where non-matching files will be moved",
    )

    args = parser.parse_args()

    print("Starting the speaker verification process")
    move_non_matching_files(args.ref, args.target, args.move)
    print("Speaker verification process completed")
