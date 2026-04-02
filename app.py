import json
from dotenv import load_dotenv
from openai import OpenAI
import chainlit as cl
from tools import scrape_video_data

load_dotenv()

client = OpenAI()

tools = [
    {
        "type": "function",
        "name": "scrape_video_data",
        "description": "Fetch detailed information about a YouTube video using its unique video ID. Returns the video title, description, tags, published date, duration, timestamped transcript, and uploading channel name.",
        "parameters": {
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "The unique identifier of the YouTube video.",
                },
            },
            "required": ["video_id"],
            "additionalProperties": False,
        },
    },
]

system_instructions = """You are YouTubeGPT, a helpful assistant designed to assist users in summarizing and answering questions about YouTube videos. You have access to a tool called scrape_video_data, which takes a YouTube Video ID as a parameter and retrieves detailed information about the video and its uploading channel. This data includes the video title, description, tags, published date, duration, timestamped transcript, and uploading channel name.

Using this tool, you can:

- Analyze the sequence of events in a video, detailing how and when key moments occur.
- Provide structured insights with timestamps, highlighting key moments and main themes.
- Enable users to explore video content efficiently without needing to watch it in full.

When a user provides a YouTube Video URL, extract the Video ID, and immediately retrieve the data using scrape_video_data and deliver a detailed summary of the video. Focus on key insights and main discussion points, weaving in relevant timestamps to support clarity and context.

After presenting the summary, invite the user to ask any follow-up questions they might have about the video or its content."""

welcome_message = """## Welcome to YouTubeGPT

YouTubeGPT allows you to ask questions and get detailed insights about a YouTube video.

- Understand the sequence of events, including how things happened and in what order.
- Get structured insights with timestamps, key moments, and main themes.
- Explore video content efficiently without watching the full video.

#### How to Use
1. Enter a YouTube video URL into the chat box.
2. YouTubeGPT will analyze and summarize it for you.
3. Ask follow-up questions to dive deeper into specific parts, or request additional details such as video metadata (title, description, tags, duration) and channel information.
"""


@cl.on_chat_start
async def start_chat():
    cl.user_session.set("conversation", [])
    await cl.Message(content=welcome_message).send()


@cl.on_message
async def handle_message(message: cl.Message):
    conversation = cl.user_session.get("conversation")
    conversation.append({"role": "user", "content": message.content})

    response = client.responses.create(
        model="gpt-5.4",
        instructions=system_instructions,
        input=conversation,
        tools=tools,
    )

    for item in response.output:
        if item.type != "function_call":
            continue

        args = json.loads(item.arguments)
        result = await cl.make_async(scrape_video_data)(args["video_id"])

        conversation.append(item)
        conversation.append({
            "type": "function_call_output",
            "call_id": item.call_id,
            "output": json.dumps(result),
        })

    reply = cl.Message(content="")
    await reply.send()

    stream = client.responses.create(
        model="gpt-5.4",
        instructions=system_instructions,
        input=conversation,
        tools=tools,
        stream=True,
    )

    for event in stream:
        if hasattr(event, "type") and event.type == "response.output_text.delta":
            await reply.stream_token(event.delta)

    await reply.update()

    conversation.append({"role": "assistant", "content": reply.content})
    cl.user_session.set("conversation", conversation)