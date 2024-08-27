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
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import re

# Import the new transcript functions
from split_and_transcribe import split_audio, transcribe_audio_chunks

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
    yt = YouTube(url)
    video_title = yt.title
    video_stream = yt.streams.get_highest_resolution()
    video_stream.download(filename="video.mp4")
    logging.info(f"Downloaded video titled: {video_title}")
    return "video.mp4", video_title

async def extract_audio_from_video(video_file_path, output_audio_path):
    logging.info(f"Extracting audio from video file: {video_file_path}")
    video_clip = VideoFileClip(video_file_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_path, codec='pcm_s16le')
    audio_clip.close()
    video_clip.close()
    logging.info(f"Audio extracted to: {output_audio_path}")

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

async def extract_transcript(audio_file_path):
    logging.info(f"Splitting audio file into chunks: {audio_file_path}")
    
    # Split the audio into smaller chunks
    chunk_files = await split_audio(audio_file_path)
    
    logging.info("Starting transcription of audio chunks...")
    
    # Transcribe the audio chunks in parallel and return the combined transcript
    transcript = await transcribe_audio_chunks(chunk_files)
    
    # Log the entire transcript
    logging.info(f"Full Transcript:\n{transcript}")
    
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
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image? Tell me about the big details and the small details. Be descriptive, but also concise."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code == 200:
        description = response.json().get('choices', [])[0].get('message', {}).get('content', 'No description found.')
        return f"{timestamp}: {description}"
    else:
        logging.error(f"Error processing image: {response.status_code}")
        return f"{timestamp}: Error {response.status_code}"

async def extract_video_visual_descriptions():
    logging.info("Extracting visual descriptions from images...")
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

def extract_channel_and_video_description(video_url):
    logging.info(f"Extracting channel name and video description from URL: {video_url}")
    
    response = requests.get(video_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    description_pattern = re.compile(r'(?<=shortDescription":").*(?=","isCrawlable)')
    video_description = description_pattern.findall(str(soup))[0].replace('\\n','\n')
    
    channel_pattern = re.compile(r'(?<=ownerChannelName":").*?(?=")')
    channel_name = channel_pattern.findall(str(soup))[0]
    
    logging.info(f"Extracted Channel Name: {channel_name}")
    logging.info(f"Extracted Description: {video_description[:60]}...")

    return channel_name, video_description

async def upload_video_data(vector_store, transcript, video_visual_descriptions, video_url, video_title, channel_name, video_description):
    logging.info("Uploading video data to Pinecone vector store...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=50
    )
    
    video_id = video_url.split("v=")[-1].split("&")[0]

    # Upload video title
    vector_store.add_texts(
        [video_title],
        metadatas=[{"video_id": video_id, "type": "video_title"}]
    )

    # Upload video description
    vector_store.add_texts(
        [video_description],
        metadatas=[{"video_id": video_id, "type": "video_description"}]
    )
    
    # Upload channel name
    vector_store.add_texts(
        [channel_name],
        metadatas=[{"video_id": video_id, "type": "channel_name"}]
    )

    # Upload transcript
    transcript_chunks = text_splitter.split_text(transcript)
    for chunk in transcript_chunks:
        vector_store.add_texts(
            [chunk], 
            metadatas=[{"video_id": video_id, "type": "transcript"}]
        )
    
    # Upload visual descriptions
    video_visual_descriptions_str = "\n".join(video_visual_descriptions)
    video_visual_description_chunks = text_splitter.split_text(video_visual_descriptions_str)
    for chunk in video_visual_description_chunks:
        vector_store.add_texts(
            [chunk], 
            metadatas=[{"video_id": video_id, "type": "video_visual_description"}]
        )
    
    logging.info("Video data upload complete")
    
    await asyncio.sleep(5)

def retrieve_context_for_querying(vector_store, video_id, context_type):
    logging.info(f"Retrieving {context_type} context for video ID: {video_id}")
    
    filter = {"video_id": video_id, "type": context_type}
    retrieved_context = vector_store.similarity_search(query="Context retrieval search", k=100, filter=filter)

    context = [chunk.page_content for chunk in retrieved_context]
    
    logging.info(f"{context_type.capitalize()} context retrieved: {len(context)} chunks found")
    return context

async def chat_with_video(video_title, transcript_context, video_visual_description_context, channel_name, video_description):
    logging.info(f"Starting chat with video: {video_title}")
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    transcript_str = "\n".join(transcript_context)
    video_visual_description_str = "\n".join(video_visual_description_context)

    system_message = f"""
    Video Title: {video_title}
    Channel Name: {channel_name}

    Video Description:
    {video_description}

    Video Transcript:
    {transcript_str}
    
    Visual Descriptions of Video:
    {video_visual_description_str}

    Your primary task is to engage in a natural, conversational manner while answering questions about a YouTube video. You will be provided with the video's title, transcript, and visual descriptions of its frames. When responding to questions, seamlessly incorporate and reason about the temporal data, such as timestamps associated with both the transcript and visual descriptions.

    For example, if a user asks what's happening at a specific time, like one minute into the video, your response should naturally weave together what is being said and what is visually occurring at that moment. Your answers should be clear, detailed, and fluid, offering a comprehensive understanding of the content at that specific timestamp.

    Additionally, if you encounter equations that are improperly formatted or incompatible with LaTeX rendering, ensure they are corrected and properly formatted using LaTeX. Use single dollar signs ($) for inline equations and double dollar signs ($$) for larger, more visual equations.
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
            messages=message_history, stream=True, model="gpt-4o", temperature=0.7
        )

        async for part in stream:
            if token := part.choices[0].delta.content or "":
                await msg.stream_token(token)

        message_history.append({"role": "assistant", "content": msg.content})
        await msg.update()

    logging.info("Chat initialized with video context")

async def does_video_exist_already(vector_store, video_id):
    logging.info(f"Checking if video already exists in the database: {video_id}")
    filter = {"video_id": video_id}
    test_query = vector_store.similarity_search(query="Chunks Existence Check", k=1, filter=filter)
    return bool(test_query)

def cleanup_files(video_file_path, audio_file_path, frames_folder, audio_chunks_folder='audio_chunks'):
    logging.info("Cleaning up files...")

    # Remove the video file if it exists
    if os.path.exists(video_file_path):
        os.remove(video_file_path)

    # Remove the audio file if it exists
    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)

    # Remove the frames folder and its contents if it exists
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)
    
    # Remove the audio chunks folder and its contents if it exists
    if os.path.exists(audio_chunks_folder):
        shutil.rmtree(audio_chunks_folder)

    logging.info("Cleanup complete")

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
6. **New Video, New Conversation**: If you wish to stop your current conversation and ask about or process a new video, simply click the "New Chat" button in the top right corner.

### Get Started
When you're ready, paste a YouTube URL into the input box to begin your interactive experience!

"""
    raw_url = await cl.AskUserMessage(content=text_content, timeout=3600).send()
    parsed_url = str(raw_url.get('output'))
    logging.info(f"User entered URL: {parsed_url}")

    video_id = extract_video_id(parsed_url)

    video_exists = await does_video_exist_already(video_data_vectorstore, video_id)

    if video_exists:
        logging.info(f"The video_id {video_id} is already in the database.")
        await cl.Message(content="Video data retrieved. Please begin your conversation.").send()
    else:
        logging.info(f"The video_id {video_id} is not in the database. Proceeding to process and upload the video data.")
        await cl.Message(content="It seems that video isn't currently in our database. Don't worry, we are currently downloading, processing, and uploading it. This will only take a few moments.").send()
        
        await asyncio.sleep(0)
        video_file_path, video_title = await download_youtube_video(parsed_url)
        channel_name, video_description = extract_channel_and_video_description(parsed_url)
        await extract_audio_from_video(video_file_path, "audio.wav")
        transcript = await extract_transcript("audio.wav")
        await extract_frames(video_file_path, "frames_output", interval_seconds=10)
        video_visual_descriptions = await extract_video_visual_descriptions()
        await upload_video_data(video_data_vectorstore, transcript, video_visual_descriptions, parsed_url, video_title, channel_name, video_description)
        cleanup_files("video.mp4", "audio.wav", "frames_output")
        await cl.Message(content="Video data processing complete! You can now start asking questions about the video.").send()
    
    # Retrieve contexts
    channel_name_context = retrieve_context_for_querying(video_data_vectorstore, video_id, "channel_name")
    video_title_context = retrieve_context_for_querying(video_data_vectorstore, video_id, "video_title")
    video_description_context = retrieve_context_for_querying(video_data_vectorstore, video_id, "video_description")
    transcript_context = retrieve_context_for_querying(video_data_vectorstore, video_id, "transcript")
    video_visual_description_context = retrieve_context_for_querying(video_data_vectorstore, video_id, "video_visual_description")

    # Convert contexts to strings
    channel_name_str = "\n".join(channel_name_context)
    video_title_str = "\n".join(video_title_context)
    video_description_str = "\n".join(video_description_context)

    # Initialize chat with video
    await chat_with_video(video_title_str, transcript_context, video_visual_description_context, channel_name_str, video_description_str)