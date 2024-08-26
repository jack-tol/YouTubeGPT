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
from langchain_pinecone import *
import asyncio
import shutil
import logging
from faster_whisper import WhisperModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def initialize_embeddings():
    logging.info("Initializing embeddings...")
    embedding = PineconeEmbeddings(model="multilingual-e5-large", batch_size=32)
    video_data_vectorstore = PineconeVectorStore.from_existing_index("all-video-data", embedding=embedding)
    return video_data_vectorstore

video_data_vectorstore = asyncio.run(initialize_embeddings())

def extract_video_id(url):
    video_id = url.split("v=")[-1].split("&")[0]
    logging.info(f"Extracted video ID: {video_id}")
    return video_id

async def download_youtube_video(url):
    logging.info(f"Downloading YouTube video from URL: {url}")
    yt = YouTube(url, use_oauth=True, allow_oauth_cache=True)
    video_title = yt.title
    video_stream = yt.streams.get_highest_resolution()
    video_stream.download(filename="video.mp4")
    logging.info(f"Downloaded video titled: {video_title}")
    return "video.mp4"

async def extract_audio_from_mp4(mp4_file_path, output_wav_file_path):
    logging.info(f"Extracting audio from video file: {mp4_file_path}")
    video_clip = VideoFileClip(mp4_file_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_wav_file_path, codec='pcm_s16le')
    audio_clip.close()
    video_clip.close()
    logging.info(f"Audio extracted to: {output_wav_file_path}")

async def extract_frames(video_path, output_folder, interval_seconds):
    logging.info(f"Extracting frames from video: {video_path}")
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
    logging.info(f"Frames extracted to folder: {output_folder}")

# Define the model outside the function
model_size = "tiny"
model = WhisperModel(model_size, device="cuda", compute_type="float32")

async def extract_transcript(audio_file):
    logging.info(f"Extracting transcript from audio file: {audio_file}")
    
    # Transcribe the audio
    segments, info = model.transcribe(audio_file, beam_size=5)
    
    def seconds_to_hhmmss(seconds):
        return str(datetime.timedelta(seconds=int(seconds)))
    
    # Initialize an empty string to store the final transcription
    transcript = ""
    
    # Aggregate the transcription from each segment with timestamps
    for segment in segments:
        start_time = seconds_to_hhmmss(segment.start)
        end_time = seconds_to_hhmmss(segment.end)
        transcript += f"{start_time} - {end_time}: {segment.text}\n"
    
    logging.info("Transcript extraction complete")
    return transcript

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def process_image(api_key, image_path):
    logging.info(f"Processing image: {image_path}")
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
                    {"type": "text", "text": "What’s in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
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
        logging.error(f"Error processing image: {response.status_code}")
        return f"{timestamp}: Error {response.status_code}"

async def extract_descriptions():
    logging.info("Extracting descriptions from images...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    
    image_folder = "frames_output"
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]
    
    descriptions = []
    
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(asyncio.run, process_image(api_key, img)): img for img in image_paths}
        
        for future in as_completed(futures):
            descriptions.append(future.result())
    
    descriptions.sort()
    
    logging.info("Image descriptions extracted")
    return descriptions

async def upload_video_data(vector_store, transcript, descriptions, video_url, video_title):
    logging.info("Uploading video data to Pinecone vector store...")
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=50
    )
    
    content_to_be_uploaded = f"{transcript}\n\n{descriptions}"
    chunks = text_splitter.split_text(content_to_be_uploaded)

    video_id = video_url.split("v=")[-1].split("&")[0]

    for chunk in chunks:
        vector_store.add_texts([chunk], metadatas=[{"video_id": video_id, "video_title": video_title}])
    
    logging.info("Video data upload complete")
    
    # Sleep to ensure data is uploaded before querying
    await asyncio.sleep(5)

def retrieve_context_for_querying(video_data_vectorstore, video_id):
    logging.info(f"Retrieving context for video ID: {video_id}")
    
    filter = {"video_id": video_id}
    retrieved_context = video_data_vectorstore.similarity_search(query="Context retrieval search", k=100, filter=filter)

    if retrieved_context:
        video_title = retrieved_context[0].metadata.get('video_title', 'Unknown Title')
    else:
        video_title = 'Unknown Title'

    context = [chunk.page_content for chunk in retrieved_context]
    
    logging.info(f"Context retrieved: {len(context)} chunks found")
    return video_title, context

async def chat_with_video(video_title, context):
    logging.info(f"Starting chat with video: {video_title}")
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context_str = "\n".join(context)

    system_message = f"""
    Video Title: {video_title}

    Retrieved Context:
    {context_str}
    
    Given the title of the video and the retrieved context, answer user questions about the video. 
    Use the timestamp information to reason about the temporal data, refer to things which were said and shown in addition to when it happened. Sometimes the equations retrieved from the context will be formatted improperly and in an incompatible format
    for correct LaTeX rendering. Therefore, if you ever need to provide equations, make sure they are
    formatted properly using LaTeX, wrapping the equation in single dollar signs ($) for inline equations
    or double dollar signs ($$) for bigger, more visual equations.
    """

    

    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": system_message}],
    )

    logging.info(f"System prompt being sent to language model:\n{system_message}")
    
    @cl.on_message
    async def handle_message(message: cl.Message):
        logging.info("User message received")
        message_history = cl.user_session.get("message_history")
        message_history.append({"role": "user", "content": message.content})

        msg = cl.Message(content="")
        await msg.send()

        stream = await client.chat.completions.create(
            messages=message_history, stream=True, model="gpt-4-turbo-2024-04-09", temperature=0.7
        )

        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await msg.stream_token(token)

        message_history.append({"role": "assistant", "content": msg.content})
        await msg.update()

    logging.info("Chat initialized with video context")

async def does_video_exist_already(video_data_vectorstore, video_id):
    logging.info(f"Checking if video already exists in the database: {video_id}")
    filter = {"video_id": video_id}
    test_query = video_data_vectorstore.similarity_search(query="Chunks Existence Check", k=1, filter=filter)
    return bool(test_query)

def cleanup_files(video_file_path, audio_file_path, frames_folder):
    logging.info("Cleaning up files...")
    if os.path.exists(video_file_path):
        os.remove(video_file_path)
    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)
    logging.info("Cleanup complete")

import asyncio

@cl.on_chat_start
async def main():
    text_content = """
## Welcome to YouTubeGPT
YouTubeGPT enables you to ask questions and get detailed answers about both the audio and visual components of a video. It understands the sequence of events, allowing you to explore how things happened and in what order, offering a complete and immersive understanding of the content.

#### Instructions
1. **Enter the URL:** Paste the YouTube video URL into the input box.
2. **Check the Database:** If the video has been processed, YouTubeGPT will instantly retrieve the data for immediate interaction.
3. **Processing New Videos:** If the video isn't in the database, YouTubeGPT will download, process, and upload the data. This may take a few moments.
4. **Ask Your Questions:** Once processed, you can ask questions about any aspect of the video, including audio, visuals, and specific moments.
5. **Reuse and Share:** Videos are processed once. Afterward, any user can instantly retrieve the data and start a conversation.

### Get Started
When you're ready, paste a YouTube URL into the input box to begin your interactive experience!

"""
    raw_url = await cl.AskUserMessage(content=text_content, timeout=3600).send()
    parsed_url = str(raw_url.get('output'))
    logging.info(f"User entered URL: {parsed_url}")

    # Extract the video ID from the URL
    video_id = extract_video_id(parsed_url)
    
    # Attempt to retrieve context for the video ID
    video_title, context = retrieve_context_for_querying(video_data_vectorstore, video_id)

    if context:
        # If video data exists, inform the user
        logging.info(f"The video_id {video_id} is already in the database.")
        await cl.Message(content="Video data retrieved. Please begin your conversation.").send()
    else:
        # If video data does not exist, immediately inform the user
        logging.info(f"The video_id {video_id} is not in the database. Proceeding to process and upload the video data.")
        await cl.Message(content="It seems that video isn't currently in our database. Don't worry, we are currently downloading, processing, and uploading it. This will only take a few moments.").send()
        
        # Begin processing the video data
        await asyncio.sleep(0)  # Allow the message to send before starting the heavy processing
        video_file_path = await download_youtube_video(parsed_url)
        video_title = YouTube(parsed_url).title

        # Extract audio from the video
        await extract_audio_from_mp4(video_file_path, "audio.wav")
        
        # Extract transcript from the audio
        transcript = await extract_transcript("audio.wav")
        
        # Extract frames from the video at a specified interval
        await extract_frames(video_file_path, "frames_output", interval_seconds=10)
        
        # Extract descriptions for each frame
        descriptions = await extract_descriptions()

        # Upload the video data to the Pinecone vector store
        await upload_video_data(video_data_vectorstore, transcript, descriptions, parsed_url, video_title)
        
        # Retrieve context again after uploading the data
        video_title, context = retrieve_context_for_querying(video_data_vectorstore, video_id)

        # Clean up the files used during the process
        cleanup_files("video.mp4", "audio.wav", "frames_output")
        
        # Inform the user that processing is complete and they can start their conversation
        await cl.Message(content="Video data processing complete! You can now start asking questions about the video.").send()
    
    # Proceed to chat with the retrieved video context
    await chat_with_video(video_title, context)
