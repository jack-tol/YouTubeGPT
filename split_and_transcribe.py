import os
from pydub import AudioSegment
from openai import OpenAI
import concurrent.futures
import logging

async def split_audio(file_path, chunk_size_mb=20):
    output_dir = 'audio_chunks'
    os.makedirs(output_dir, exist_ok=True)
    
    audio = AudioSegment.from_file(file_path)
    total_size = len(audio.raw_data)  # Size in bytes
    total_duration = len(audio)  # Duration in milliseconds
    
    chunk_size_bytes = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
    chunk_duration_ms = int((chunk_size_bytes / total_size) * total_duration)
    
    chunk_files = []
    for i in range(0, len(audio), chunk_duration_ms):
        chunk = audio[i:i + chunk_duration_ms]
        chunk_filename = os.path.join(output_dir, f'chunk_{i // chunk_duration_ms}.wav')
        chunk.export(chunk_filename, format="wav")
        chunk_files.append((chunk_filename, i))  # Save file and start time
        print(f'Created: {chunk_filename}')
    
    return chunk_files

def transcribe_chunk(client, chunk_file, start_offset_ms):
    logging.info(f"Transcribing chunk: {chunk_file}")

    with open(chunk_file, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file, 
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    
    transcription_dict = transcription.to_dict()
    segments = transcription_dict['segments']
    
    adjusted_segments = []
    for segment in segments:
        start_time = segment['start'] + start_offset_ms / 1000  # Convert to seconds
        end_time = segment['end'] + start_offset_ms / 1000
        text = segment['text']
        start_time_formatted = f"{int(start_time//3600):02}:{int((start_time%3600)//60):02}:{int(start_time%60):02}"
        end_time_formatted = f"{int(end_time//3600):02}:{int((end_time%3600)//60):02}:{int(end_time%60):02}"
        adjusted_segments.append((start_time, f"{start_time_formatted} - {end_time_formatted} - {text}"))
    
    logging.info(f"Finished transcribing chunk: {chunk_file}")
    
    return adjusted_segments


async def transcribe_audio_chunks(chunk_files):
    client = OpenAI()

    def transcribe_and_collect(chunk_file, start_offset):
        return transcribe_chunk(client, chunk_file, start_offset)

    all_transcriptions = []

    # Use ThreadPoolExecutor to run transcription in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(transcribe_and_collect, chunk_file, start_offset)
            for chunk_file, start_offset in chunk_files
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                transcription = future.result()
                all_transcriptions.extend(transcription)
            except Exception as exc:
                print(f'Error during transcription: {exc}')
    
    # Sort transcriptions by start time to ensure correct order
    all_transcriptions.sort(key=lambda x: x[0])
    
    # Combine the ordered transcriptions into a single string
    transcript = "\n".join(transcription for _, transcription in all_transcriptions)
    
    return transcript
