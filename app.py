import uuid
from pytubefix import YouTube
import chainlit as cl
import aiofiles
import aiohttp
from moviepy.editor import VideoFileClip
import cv2
import os
import base64
from openai import AsyncOpenAI
from langchain_pinecone import *
import asyncio
import shutil
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tenacity import retry, stop_after_attempt, wait_random_exponential
import re

from split_and_transcribe import split_audio, transcribe_audio_chunks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def initialize_embeddings():
    logging.info("Initializing embeddings...")
    embedding = PineconeEmbeddings(model="multilingual-e5-large", batch_size=32)
    video_data_vectorstore = PineconeVectorStore.from_existing_index("all-video-data", embedding=embedding)
    return video_data_vectorstore

video_data_vectorstore = asyncio.run(initialize_embeddings())

def is_valid_youtube_url(url):
    youtube_regex = (
        r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/"
        r"(watch\?v=|embed/|v/|shorts/|youtu\.be/|.+\?v=)?([^&=%\?]{11})"
    )
    match = re.match(youtube_regex, url)
    return bool(match)


def extract_video_id(url):
    # Regex patterns for different YouTube URL formats including Shorts
    patterns = [
        r"(?:v=|v/|embed/|youtu\.be/|watch\?v=|be/|youtu\.be/)([^&=%\?]{11})",  # Regular YouTube URLs
        r"(?:youtube\.com/shorts/)([^&=%\?]{11})"  # YouTube Shorts URLs
    ]
    
    video_id = None
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            video_id = match.group(1)
            break
    
    if video_id:
        logging.info(f"Extracted video ID: {video_id}")
        return video_id
    else:
        logging.error(f"Could not extract video ID from URL: {url}")
        return None


async def download_youtube_video(url, session_id):
    logging.info(f"[{session_id}] Downloading YouTube video from URL: {url}")
    
    yt = YouTube(url, client='MWEB', use_oauth=True, allow_oauth_cache=True, token_file="tokens.json")
    
    video_title = yt.title
    video_stream = yt.streams.get_highest_resolution()
    video_path = f"{session_id}_video.mp4"
    
    await asyncio.to_thread(video_stream.download, filename=video_path)
    logging.info(f"[{session_id}] Downloaded video titled: {video_title}")
    
    return video_path, video_title

async def extract_audio_from_video(video_file_path, session_id):
    output_audio_path = f"{session_id}_audio.wav"
    logging.info(f"[{session_id}] Extracting audio from video file: {video_file_path}")
    await asyncio.to_thread(extract_audio, video_file_path, output_audio_path)
    logging.info(f"[{session_id}] Audio extracted to: {output_audio_path}")
    return output_audio_path

def extract_audio(video_file_path, output_audio_path):
    video_clip = VideoFileClip(video_file_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_path, codec='pcm_s16le')
    audio_clip.close()
    video_clip.close()

async def extract_frames(video_path, session_id, interval_seconds):
    output_folder = f"{session_id}_frames_output"
    logging.info(f"[{session_id}] Extracting frames from video: {video_path}")
    await asyncio.to_thread(extract_frames_sync, video_path, output_folder, interval_seconds)
    logging.info(f"[{session_id}] Frames extracted to folder: {output_folder}")

def extract_frames_sync(video_path, output_folder, interval_seconds):
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

async def extract_transcript(audio_file_path, session_id):
    logging.info(f"[{session_id}] Splitting audio file into chunks: {audio_file_path}")
    
    # Ensure session_id is passed to split_audio
    chunk_files = await split_audio(audio_file_path, session_id)
    
    logging.info(f"[{session_id}] Starting transcription of audio chunks...")
    
    # Ensure session_id is passed to transcribe_audio_chunks
    transcript = await transcribe_audio_chunks(chunk_files, session_id)
    
    logging.info(f"[{session_id}] Transcript extraction complete")
    return transcript

async def encode_image(image_path):
    async with aiofiles.open(image_path, "rb") as image_file:
        return base64.b64encode(await image_file.read()).decode('utf-8')

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def process_image(api_key, image_path, session_id):
    logging.info(f"[{session_id}] Processing image: {image_path}")
    base64_image = await encode_image(image_path)
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
                    {"type": "text", "text": "What's in this image? Tell me about the big details and the small details..."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ]
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as response:
            if response.status == 200:
                description = (await response.json()).get('choices', [])[0].get('message', {}).get('content', 'No description found.')
                return f"{timestamp}: {description}"
            else:
                logging.error(f"[{session_id}] Error processing image: {response.status}")
                return f"{timestamp}: Error {response.status}"


async def extract_video_visual_descriptions(session_id):
    logging.info(f"[{session_id}] Extracting visual descriptions from images...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    
    image_folder = f"{session_id}_frames_output"
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.jpg', '.jpeg', '.png'))]
    
    descriptions = []
    
    tasks = [process_image(api_key, img, session_id) for img in image_paths]
    
    descriptions = await asyncio.gather(*tasks)
    
    descriptions.sort()
    
    logging.info(f"[{session_id}] Image descriptions extracted")
    return descriptions

import os
import logging
import googleapiclient.discovery
import re

async def extract_channel_and_video_description(video_url, session_id):
    logging.info(f"[{session_id}] Extracting channel name and video description from URL: {video_url}")
    
    # Initialize channel name and video description
    channel_name = ""
    video_description = ""

    try:
        # Fetch the API key from environment variables
        api_key = os.getenv('YOUTUBE_API_KEY')
        if not api_key:
            raise ValueError("YouTube API key is not set in environment variables.")

        # Function to get video ID from URL
        def get_video_id(youtube_url):
            video_id_match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', youtube_url)
            if video_id_match:
                return video_id_match.group(1)
            else:
                raise ValueError("Invalid YouTube URL")

        # Extract the video ID
        video_id = get_video_id(video_url)
        
        # Build the YouTube API client
        youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=api_key)
        
        # Make the API request to get video details
        video_request = youtube.videos().list(part="snippet", id=video_id)
        video_response = video_request.execute()
        
        # Extract the video details
        video_snippet = video_response['items'][0]['snippet']
        channel_name = video_snippet['channelTitle']
        video_description = video_snippet['description']
        
        logging.info(f"[{session_id}] Extracted Channel Name: {channel_name}")
        logging.info(f"[{session_id}] Extracted Video Description: {video_description[:60]}...")
        
    except Exception as e:
        logging.error(f"[{session_id}] Error extracting video description and channel name: {e}")

    return channel_name, video_description

async def upload_video_data(vector_store, transcript, video_visual_descriptions, video_title, channel_name, video_description, video_id, session_id):
    logging.info(f"[{session_id}] Uploading video data to Pinecone vector store...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000,
        chunk_overlap=0
    )

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

    # Upload transcript chunks
    transcript_chunks = text_splitter.split_text(transcript)
    for chunk in transcript_chunks:
        vector_store.add_texts(
            [chunk], 
            metadatas=[{"video_id": video_id, "type": "transcript"}]
        )
    
    # Process and upload video visual descriptions in batches
    video_visual_descriptions_str = "\n".join(video_visual_descriptions)
    video_visual_description_chunks = text_splitter.split_text(video_visual_descriptions_str)

    batch_size = 10  # Adjust batch size as necessary
    for i in range(0, len(video_visual_description_chunks), batch_size):
        batch = video_visual_description_chunks[i:i + batch_size]
        vector_store.add_texts(
            batch, 
            metadatas=[{"video_id": video_id, "type": "video_visual_description"}]
        )

        # Yield control to allow other tasks to run
        await asyncio.sleep(0)

    logging.info(f"[{session_id}] Video data upload complete")

def cleanup_files(session_id):
    logging.info(f"[{session_id}] Cleaning up files...")

    video_file_path = f"{session_id}_video.mp4"
    if os.path.exists(video_file_path):
        os.remove(video_file_path)

    audio_file_path = f"{session_id}_audio.wav"
    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)

    frames_folder = f"{session_id}_frames_output"
    if os.path.exists(frames_folder):
        shutil.rmtree(frames_folder)
    
    # Clean up the session-specific directory if it exists
    session_dir = f"{session_id}"
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
        logging.info(f"[{session_id}] Cleaned up session directory.")

    logging.info(f"[{session_id}] Cleanup complete")

async def does_video_exist_already(vector_store, video_id, session_id):
    logging.info(f"[{session_id}] Checking if video already exists in the database: {video_id}")
    filter = {"video_id": video_id}
    test_query = vector_store.similarity_search(query="Chunks Existence Check", k=1, filter=filter)
    return bool(test_query)

async def retrieve_context_for_querying(vector_store, video_id, context_type, session_id):
    logging.info(f"[{session_id}] Retrieving {context_type} context for video ID: {video_id}")
    
    filter = {"video_id": video_id, "type": context_type}
    retrieved_context = vector_store.similarity_search(query="Context retrieval search", k=100, filter=filter)

    context = [chunk.page_content for chunk in retrieved_context]
    
    logging.info(f"[{session_id}] {context_type.capitalize()} context retrieved: {len(context)} chunks found")
    return context

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def chat_with_video(session_id, video_title, transcript_context, video_visual_description_context, channel_name, video_description):
    logging.info(f"[{session_id}] Starting chat with video: {video_title}")
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

    Your primary task is to engage in a natural, conversational manner while answering questions about a YouTube video...
    """

    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": system_message}],
    )

    @cl.on_message
    async def handle_message(message: cl.Message):
        logging.info(f"[{session_id}] User message received")
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

    logging.info(f"[{session_id}] Chat initialized with video context")


@cl.on_chat_start
async def main():
    session_id = str(uuid.uuid4())

    # Welcome message displayed once at the start
    welcome_message = """
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

    # Message to prompt the user for a valid input
    prompt_message = "Please enter a valid YouTube URL."

    # Display the welcome message and prompt for the first time
    raw_url = await cl.AskUserMessage(content=welcome_message, timeout=3600).send()

    while True:
        # Check if raw_url is None or if there's no output
        if raw_url is None or not raw_url.get('output'):
            await cl.Message(content="No input received. Please enter a YouTube URL:").send()
            raw_url = await cl.AskUserMessage(content=prompt_message, timeout=3600).send()
            continue

        parsed_url = str(raw_url.get('output'))
        logging.info(f"[{session_id}] User entered URL: {parsed_url}")

        # Validate the YouTube URL and extract the video_id
        video_id = extract_video_id(parsed_url)

        if not video_id:
            # Prompt the user to enter a valid YouTube URL
            raw_url = await cl.AskUserMessage(content=prompt_message, timeout=3600).send()
            continue  # Go back to asking for user input with the prompt message

        # Construct the full YouTube URL using the extracted video_id
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        logging.info(f"[{session_id}] Constructed full YouTube URL: {video_url}")

        # Check if video already exists in the database
        logging.info(f"[{session_id}] Checking if video already exists in the database: {video_id}")
        video_exists = await does_video_exist_already(video_data_vectorstore, video_id, session_id)

        if video_exists:
            logging.info(f"[{session_id}] The video_id {video_id} is already in the database.")
            
            # Update UI: Video already processed, retrieving context
            status_msg = cl.Message(content="**Video already processed. Retrieving context...**")
            await status_msg.send()

            # Force the event loop to process the UI update before retrieving context
            await asyncio.sleep(0)
            
            # Retrieve context sequentially
            channel_name_context = await retrieve_context_for_querying(video_data_vectorstore, video_id, "channel_name", session_id)
            video_title_context = await retrieve_context_for_querying(video_data_vectorstore, video_id, "video_title", session_id)
            video_description_context = await retrieve_context_for_querying(video_data_vectorstore, video_id, "video_description", session_id)
            transcript_context = await retrieve_context_for_querying(video_data_vectorstore, video_id, "transcript", session_id)
            video_visual_description_context = await retrieve_context_for_querying(video_data_vectorstore, video_id, "video_visual_description", session_id)

            # Update UI: Video data retrieved, user can now start asking questions
            status_msg.content = "**Video data retrieved. You can now start asking questions.**"
            await status_msg.update()

        else:
            logging.info(f"[{session_id}] The video_id {video_id} is not in the database. Proceeding to process and upload the video data.")
            
            # Update 1: Video is being downloaded
            status_msg = cl.Message(content="**It seems that video isn't currently in our database. The video is being downloaded...**")
            await status_msg.send()
            
            await asyncio.sleep(0)

            # Download the video
            video_file_path, video_title = await download_youtube_video(video_url, session_id)
            logging.info(f"[{session_id}] Video downloaded successfully.")

            # Update 2: Video downloaded, proceeding with metadata extraction
            status_msg.content = "**Video downloaded successfully. Proceeding with metadata extraction...**"
            await status_msg.update()

            # Extract metadata
            channel_name, video_description = await extract_channel_and_video_description(video_url, session_id)
            logging.info(f"[{session_id}] Metadata extracted.")

            # Update 3: Metadata extracted, audio processing underway
            status_msg.content = "**Metadata extracted. Audio processing underway...**"
            await status_msg.update()

            # Extract audio and process transcript
            audio_file_path = await extract_audio_from_video(video_file_path, session_id)
            transcript = await extract_transcript(audio_file_path, session_id)
            logging.info(f"[{session_id}] Audio processing completed.")

            # Update 4: Audio processing completed, starting visual processing
            status_msg.content = "**Audio processing completed. Starting visual processing...**"
            await status_msg.update()

            # Extract frames and process image descriptions
            await extract_frames(video_file_path, session_id, interval_seconds=10)
            video_visual_descriptions = await extract_video_visual_descriptions(session_id)
            logging.info(f"[{session_id}] Visual processing completed.")

            # Update 5: Visual processing completed, proceeding to upload the data
            status_msg.content = "**Visual processing completed. Proceeding to upload the data...**"
            await status_msg.update()

            # Yield control back to the event loop to ensure the UI update is processed
            await asyncio.sleep(0)

            # Upload video data
            await upload_video_data(video_data_vectorstore, transcript, video_visual_descriptions, video_title, channel_name, video_description, video_id, session_id)
            await asyncio.sleep(5)
            logging.info(f"[{session_id}] Upload complete.")

            # Update 6: Upload complete, retrieving video data
            status_msg.content = "**Upload complete. Retrieving video data...**"
            await status_msg.update()

            cleanup_files(session_id)

            # Notify the user that context retrieval is starting
            logging.info(f"[{session_id}] Retrieving video data context.")
            status_msg.content = "**Retrieving video data context...**"
            await status_msg.update()

            # Retrieve context sequentially
            channel_name_context = await retrieve_context_for_querying(video_data_vectorstore, video_id, "channel_name", session_id)
            video_title_context = await retrieve_context_for_querying(video_data_vectorstore, video_id, "video_title", session_id)
            video_description_context = await retrieve_context_for_querying(video_data_vectorstore, video_id, "video_description", session_id)
            transcript_context = await retrieve_context_for_querying(video_data_vectorstore, video_id, "transcript", session_id)
            video_visual_description_context = await retrieve_context_for_querying(video_data_vectorstore, video_id, "video_visual_description", session_id)

            # Update UI: Video data retrieved, user can now start asking questions
            status_msg.content = "**Video data retrieved. You can now start asking questions.**"
            await status_msg.update()

        # Prepare contexts for the conversation
        channel_name_str = "\n".join(channel_name_context)
        video_title_str = "\n".join(video_title_context)
        video_description_str = "\n".join(video_description_context)

        # Start the chat with the video context
        await chat_with_video(session_id, video_title_str, transcript_context, video_visual_description_context, channel_name_str, video_description_str)

        break