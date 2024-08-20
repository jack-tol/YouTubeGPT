import whisper
import datetime
from pytubefix import YouTube
import chainlit as cl
from moviepy.editor import VideoFileClip
import cv2
import os
import base64
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# Function to download the YouTube video
def download_youtube_video(url):
    yt = YouTube(url)
    video_title = yt.title
    print("Title:")
    print(video_title)
    ys = yt.streams.get_highest_resolution()
    ys.download(filename="video.mp4")
    return "video.mp4"

# Function to extract audio from the video
def extract_audio_from_mp4(mp4_file_path, output_wav_file_path):
    video_clip = VideoFileClip(mp4_file_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_wav_file_path, codec='pcm_s16le')
    audio_clip.close()
    video_clip.close()

# Function to extract frames from the video at specified intervals
def extract_frames(video_path, output_folder, interval_seconds):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    frame_count = 0
    success = True

    while success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        success, frame = cap.read()

        if success:
            total_seconds = int(frame_count / fps)
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            formatted_time = f"{hours:02}_{minutes:02}_{seconds:02}"
            filename = os.path.join(output_folder, f"{formatted_time}.jpg")
            cv2.imwrite(filename, frame)
        
        frame_count += frame_interval

    cap.release()

# Function to extract transcript from audio using Whisper
def extract_transcript(audio_file):
    model = whisper.load_model("tiny")
    
    result = model.transcribe(audio_file)
    
    def seconds_to_hhmmss(seconds):
        return str(datetime.timedelta(seconds=int(seconds)))
    
    transcript = ""
    for segment in result['segments']:
        start_time = seconds_to_hhmmss(segment['start'])
        transcript += f"{start_time} - {segment['text']}\n"
    
    return transcript

# Function to encode an image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to process a single image and extract a description
def process_image(api_key, image_path):
    base64_image = encode_image(image_path)
    timestamp = os.path.basename(image_path).split('.')[0]  # Assuming the filename is in HH_MM_SS format
    
    # Replace underscores with colons to match the HH:MM:SS format
    timestamp = timestamp.replace('_', ':')
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What’s in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 50
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code == 200:
        description = response.json().get('choices', [])[0].get('message', {}).get('content', 'No description found.')
        return f"{timestamp}: {description}"
    else:
        return f"{timestamp}: Error {response.status_code}"

# Function to extract descriptions from all images in the 'frames_output' folder
def extract_descriptions():
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    
    # Define image folder and paths
    image_folder = "frames_output"
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]
    
    descriptions = []
    
    # Process images concurrently
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, api_key, img): img for img in image_paths}
        
        for future in as_completed(futures):
            descriptions.append(future.result())
    
    descriptions.sort()  # Sort descriptions by timestamp
    
    # Join all descriptions with line breaks and print them
    output = "\n".join(descriptions)
    print(output)
    
    return descriptions

@cl.on_chat_start
async def main():
    raw_url = await cl.AskUserMessage(content="Please Enter the URL of the YouTube Video You Wish to Interact With:", timeout=3600).send()
    parsed_url = str(raw_url.get('output'))
    video_file_path = download_youtube_video(parsed_url)
    extract_audio_from_mp4(video_file_path, "audio.wav")
    
    # Extract transcript from the audio
    transcript = extract_transcript("audio.wav")
    print("Transcript:")
    print(transcript)
    
    # Extract frames from the video
    extract_frames(video_file_path, "frames_output", interval_seconds=10)
    
    # Extract descriptions from the frames
    descriptions = extract_descriptions()
    print("Descriptions:")
    print("\n".join(descriptions))
    
    await cl.Message(content="Download complete, audio extracted, transcript generated, frames extracted, and descriptions extracted!").send()