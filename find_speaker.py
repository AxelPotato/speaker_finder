import os
import argparse
import logging
import coloredlogs
from pathlib import Path
from pydub import AudioSegment
from resemblyzer import preprocess_wav, VoiceEncoder
from scipy.spatial.distance import cosine

# Set up logger
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger, fmt='%(asctime)s %(levelname)s %(message)s')

def load_wav(file_path):
    temp_wav_file = None
    logger.info(f"Loading audio file: {file_path}")
    # Convert MP3 to WAV if needed
    if file_path.endswith(".mp3"):
        logger.info(f"Converting MP3 to WAV for file: {file_path}")
        audio = AudioSegment.from_mp3(file_path)
        temp_wav_file = file_path.replace(".mp3", ".wav")
        audio.export(temp_wav_file, format="wav")
        file_path = temp_wav_file
    
    wav = preprocess_wav(file_path)
    return wav, temp_wav_file

def process_files(reference_file, target_folder, new_folder, similarity_threshold=0.88):
    # Load the reference audio file
    logger.info(f"Loading reference audio file: {reference_file}")
    reference_wav, _ = load_wav(reference_file)
    
    # Initialize the voice encoder
    logger.info("Initializing the voice encoder")
    encoder = VoiceEncoder()

    # Embed the reference audio
    logger.info("Embedding the reference audio")
    reference_embed = encoder.embed_utterance(reference_wav)

    # Ensure the new folder exists
    Path(new_folder).mkdir(parents=True, exist_ok=True)

    # Process each file in the target folder recursively
    logger.info(f"Processing files in the target folder recursively: {target_folder}")
    for root, _, files in os.walk(target_folder):
        for audio_file in files:
            audio_file_path = os.path.join(root, audio_file)

            # Check file extension and skip non-audio files
            if not (audio_file_path.endswith(".wav") or audio_file_path.endswith(".mp3")):
                logger.warning(f"Skipping non-audio file: {audio_file}")
                continue

            # Load and embed the target audio file
            temp_wav_file = None
            try:
                logger.info(f"Loading and embedding target audio file: {audio_file_path}")
                target_wav, temp_wav_file = load_wav(audio_file_path)
                target_embed = encoder.embed_utterance(target_wav)

                # Calculate similarity using cosine distance
                similarity = 1 - cosine(reference_embed, target_embed)
                if similarity < similarity_threshold:
                    similarity_msg = f"\033[1m\033[91m{similarity:.4f}\033[0m"  # Red color for below threshold
                    logger.info(f"Similarity for {audio_file}: {similarity_msg}")
                else:
                    similarity_msg = f"\033[1m\033[94m{similarity:.4f}\033[0m"  # Blue color for above threshold
                    logger.info(f"Similarity for {audio_file}: {similarity_msg}")
                    logger.info(f"Copying {audio_file} to {new_folder} due to high similarity")
                    shutil.copy(audio_file_path, os.path.join(new_folder, audio_file))
            except Exception as e:
                logger.error(f"Could not process {audio_file}: {e}")
            finally:
                # Clean up temporary WAV file if created
                if temp_wav_file and os.path.exists(temp_wav_file):
                    logger.info(f"Deleting temporary file: {temp_wav_file}")
                    os.remove(temp_wav_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speaker verification script.")
    parser.add_argument("--ref", type=str, required=True, help="Path to the reference audio file")
    parser.add_argument("--target", type=str, required=True, help="Path to the folder containing audio files to check")
    parser.add_argument("--new", type=str, required=True, help="Path to the folder where matching files will be copied")

    args = parser.parse_args()

    logger.info("Starting the speaker verification process")
    process_files(args.ref, args.target, args.new)
    logger.info("Speaker verification process completed")
