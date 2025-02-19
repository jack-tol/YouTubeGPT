import chainlit as cl
import requests
import re
from prompt_template import generate_prompt_template
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def is_valid_youtube_url(url):
    youtube_regex = re.compile(
        r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/"
        r"(watch\?v=|embed/|v/|shorts/|youtu\.be/)?([^&/%?=]{11})"
    )

    match = youtube_regex.search(url)
    return bool(match)

async def fetch_video_data(video_url):
    response = requests.get(f"https://youtubegpt-api.fly.dev/get_video_data?video_url={video_url}")
    response_json = response.json()
    print(f"Retrieved video data for the video title '{response_json.get('video_name', 'Unknown Title')}' by Channel '{response_json.get('channel_name', 'Unknown Channel')}'")
    return response_json

async def build_prompt(data):
    return generate_prompt_template(data)

async def query_openai(prompt):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": prompt})
    cl.user_session.set("message_history", message_history)

    msg = cl.Message(content="")
    completion_stream = await client.chat.completions.create(model="gpt-4o", messages=message_history, stream=True)

    assistant_response = ""
    async for part in completion_stream:
        if part.choices and part.choices[0].delta:
            token = part.choices[0].delta.content
            if token:
                assistant_response += token
                await msg.stream_token(token)

    await msg.update()
    message_history.append({"role": "assistant", "content": assistant_response})
    cl.user_session.set("message_history", message_history)

@cl.on_chat_start
async def welcome_message():
    cl.user_session.set("message_history", [])
    cl.user_session.set("video_mode", False)

    await cl.Message(content="""### Welcome to YouTubeGPT  

YouTubeGPT allows you to ask questions and get detailed insights about a YouTube video.  

- Understand the sequence of events, including how things happened and in what order.  
- Get structured insights with timestamps, key moments, and main themes.  
- Explore video content efficiently without watching the full video.  

#### How to Use  
1. Enter a YouTube video URL into the chat box.  
2. YouTubeGPT will analyze and summarize it for you.  
3. Ask follow-up questions to dive deeper into specific parts, or request additional details such as video metadata (title, description, tags, duration) and channel information.  
""").send()

@cl.on_message
async def handle_user_input(message: cl.Message):
    video_mode = cl.user_session.get("video_mode", False)
    
    if not video_mode:
        video_url = message.content.strip()

        if not is_valid_youtube_url(video_url):
            await cl.Message(content="**Invalid URL!** Please enter a valid **YouTube video URL.**").send()
            return

        video_data = await fetch_video_data(video_url)
        cl.user_session.set("video_mode", True)

        prompt_template = await build_prompt(video_data)
        await query_openai(prompt_template)

    else:
        await query_openai(message.content)