import os
from pydub import AudioSegment
from datetime import timedelta
import logging
import aiohttp
import asyncio

API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = "https://api.openai.com/v1/audio/transcriptions"

# Function to convert seconds to HH:MM:SS format
def format_time(seconds):
    return str(timedelta(seconds=round(seconds)))

# Asynchronous function to process a single audio chunk
async def process_chunk(file_info, session_id):
    file_path, chunk_start_time = file_info
    headers = {
        "Authorization": f"Bearer {API_KEY}"
    }
    
    # Use FormData to handle file upload
    data = aiohttp.FormData()
    data.add_field('model', 'whisper-1')
    data.add_field('response_format', 'verbose_json')
    data.add_field('timestamp_granularities', 'segment')
    data.add_field('file', open(file_path, 'rb'), filename=os.path.basename(file_path), content_type='audio/wav')
    
    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=headers, data=data) as response:
            if response.status == 200:
                transcript = await response.json()
            else:
                logging.error(f"[{session_id}] Error in transcription request: {response.status}")
                return [], chunk_start_time  # Return an empty list if there was an error
    
    adjusted_segments = []
    for segment in transcript.get('segments', []):
        # Adjust the start and end time based on the chunk's start time
        segment['start'] += chunk_start_time
        segment['end'] += chunk_start_time
        adjusted_segments.append(segment)
    
    logging.info(f"[{session_id}] Processed chunk starting at {format_time(chunk_start_time)}")
    return adjusted_segments, chunk_start_time

# Function to format the transcript segments
def format_transcript(segments, session_id):
    final_transcript = ""
    for segment in segments:
        start_time = format_time(segment['start'])
        end_time = format_time(segment['end'])
        text = segment['text']
        final_transcript += f"{start_time} - {end_time} - {text}\n"
    logging.info(f"[{session_id}] Full Transcript Generated")
    return final_transcript

# Function to split the audio file into chunks
async def split_audio(file_path, session_id, chunk_size_mb=20):
    output_dir = os.path.join(session_id, 'audio_chunks')
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
        chunk_files.append((chunk_filename, i // 1000))  # Save file and start time in seconds
        logging.info(f'[{session_id}] Created audio chunk: {chunk_filename}')
    
    return chunk_files

# Main function to process the audio file
async def transcribe_audio_chunks(chunk_files, session_id):
    # Step 1: Transcribe the audio chunks in parallel using asyncio.gather
    tasks = [process_chunk(file_info, session_id) for file_info in chunk_files]
    results = await asyncio.gather(*tasks)
    
    # Step 2: Sort all transcripts based on chunk start time
    all_transcripts = sorted(results, key=lambda x: x[1])
    
    # Step 3: Concatenate and format the final transcript
    final_segments = []
    for segments, _ in all_transcripts:
        final_segments.extend(segments)
    
    final_transcript = format_transcript(final_segments, session_id)

    return final_transcript