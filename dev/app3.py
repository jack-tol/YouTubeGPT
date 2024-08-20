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
from openai import AsyncOpenAI

def download_youtube_video(url):
    yt = YouTube(url,
        use_oauth=True,
        allow_oauth_cache=True)
    video_title = yt.title
    print("Title:")
    print(video_title)
    ys = yt.streams.get_highest_resolution()
    ys.download(filename="video.mp4")
    return "video.mp4"

def extract_audio_from_mp4(mp4_file_path, output_wav_file_path):
    video_clip = VideoFileClip(mp4_file_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_wav_file_path, codec='pcm_s16le')
    audio_clip.close()
    video_clip.close()

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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image(api_key, image_path):
    base64_image = encode_image(image_path)
    timestamp = os.path.basename(image_path).split('.')[0]
    
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

def extract_descriptions():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    
    image_folder = "frames_output"
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]
    
    descriptions = []
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, api_key, img): img for img in image_paths}
        
        for future in as_completed(futures):
            descriptions.append(future.result())
    
    descriptions.sort()
    
    output = "\n".join(descriptions)
    print(output)
    
    return descriptions

async def chat_with_video(video_title, transcript, descriptions):
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context = f"""
    Video Title: {video_title}
    Transcript of the Video:
    {transcript}

    Descriptions of the Visual Content:
    {descriptions}
    
    Given the title of the video, transcript of the video, and descriptions of the visual content, answer user questions about the video. 
    Use the timestamp information to reason about the temporal data, refer to things which were said and shown in addition to when it happened. Sometimes the equations retrieved from the context will be formatted improperly and in an incompatible format
         for correct LaTeX rendering. Therefore, if you ever need to provide equations, make sure they are
         formatted properly using LaTeX, wrapping the equation in single dollar signs ($) for inline equations
         or double dollar signs ($$) for bigger, more visual equations.
    """

    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": context}],
    )

    @cl.on_message
    async def handle_message(message: cl.Message):
        message_history = cl.user_session.get("message_history")
        message_history.append({"role": "user", "content": message.content})

        msg = cl.Message(content="")
        await msg.send()

        stream = await client.chat.completions.create(
            messages=message_history, stream=True, model="gpt-4o", temperature=0.7
        )

        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await msg.stream_token(token)

        message_history.append({"role": "assistant", "content": msg.content})
        await msg.update()

@cl.on_chat_start
async def main():
    raw_url = await cl.AskUserMessage(content="Please Enter the URL of the YouTube Video You Wish to Interact With:", timeout=3600).send()
    parsed_url = str(raw_url.get('output'))
    video_file_path = download_youtube_video(parsed_url)
    video_title = YouTube(parsed_url).title

    extract_audio_from_mp4(video_file_path, "audio.wav")
    
    transcript = extract_transcript("audio.wav")
    print("Transcript:")
    print(transcript)
    
    extract_frames(video_file_path, "frames_output", interval_seconds=10)
    
    descriptions = extract_descriptions()
    print("Descriptions:")
    print("\n".join(descriptions))
    
    await chat_with_video(video_title, transcript, descriptions)
    
    await cl.Message(content="Download complete, audio extracted, transcript generated, frames extracted, and descriptions extracted! You can now start asking questions about the video.").send()