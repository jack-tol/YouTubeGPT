import asyncio
import json
from openai import AsyncOpenAI
import aiohttp
import chainlit as cl

tool_schemas = [
    {
        "type": "function",
        "function": {
            "name": "fetch_video_data",
            "description": "Fetch detailed information about a YouTube video using its unique video ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_id": {
                        "type": "string",
                        "description": "The unique identifier of the YouTube video."
                    }
                },
                "required": ["video_id"],
                "additionalProperties": False
            }
        }
    }
]

async def fetch_video_data(video_id):
    url = f"https://youtubegpt-api.fly.dev/get_video_data?video_url=https://www.youtube.com/watch?v={video_id}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            video_data = await response.json()
            print(f"Retrieved video data for the video title '{video_data.get('video_name', 'Unknown Title')}' by Channel '{video_data.get('channel_name', 'Unknown Channel')}'")
            return video_data

client = AsyncOpenAI()
MODEL_TEMPERATURE = 0.2

tool_functions = {
    "fetch_video_data": fetch_video_data
}

async def call_gpt(message_history, assistant_message):
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=message_history,
        tools=tool_schemas,
        tool_choice="auto",
        stream=True,
        temperature=MODEL_TEMPERATURE,
    )
    tool_calls = []
    assistant_content = ""
    async for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content is not None:
            assistant_content += delta.content
            assistant_message.content += delta.content
            await assistant_message.update()
        for tool_call_delta in delta.tool_calls or []:
            index = tool_call_delta.index
            existing_tool_call = next((tc for tc in tool_calls if tc['index'] == index), None)
            if existing_tool_call is None:
                tool_calls.append({
                    'index': index,
                    'id': tool_call_delta.id,
                    'name': tool_call_delta.function.name,
                    'arguments': tool_call_delta.function.arguments or ''
                })
            else:
                if tool_call_delta.function.arguments:
                    existing_tool_call['arguments'] += tool_call_delta.function.arguments
        if choice.finish_reason:
            finish_reason = choice.finish_reason

    if finish_reason == 'tool_calls' and tool_calls:
        for tc in tool_calls:
            print(f"Tool to call: {tc['name']} with arguments: {tc['arguments']}")
        assistant_message_history = {
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": [
                {
                    "id": tc['id'],
                    "type": "function",
                    "function": {
                        "name": tc['name'],
                        "arguments": tc['arguments']
                    }
                } for tc in tool_calls
            ]
        }
        message_history.append(assistant_message_history)

        async def execute_tool(tool_call):
            try:
                function_name = tool_call['name']
                arguments = json.loads(tool_call['arguments'])
                function = tool_functions[function_name]
                print(f"Executing tool: {function_name}")
                result = await function(**arguments)
                return result
            except Exception as e:
                return f"Error executing {function_name}: {str(e)}"

        tool_results = await asyncio.gather(*[execute_tool(tc) for tc in tool_calls])
        for tool_call, result in zip(tool_calls, tool_results):
            message_history.append({
                "role": "tool",
                "tool_call_id": tool_call['id'],
                "content": str(result)
            })

        assistant_message.content += "\n\n"
        await assistant_message.update()
        final_stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=message_history,
            stream=True,
            temperature=MODEL_TEMPERATURE,
        )
        final_assistant_content = ""
        async for chunk in final_stream:
            if chunk.choices[0].delta.content is not None:
                final_content = chunk.choices[0].delta.content
                assistant_message.content += final_content
                await assistant_message.update()
                final_assistant_content += final_content
        message_history.append({"role": "assistant", "content": final_assistant_content})
    elif finish_reason == 'stop':
        message_history.append({"role": "assistant", "content": assistant_content})
    else:
        print(f"Unexpected finish reason: {finish_reason}")

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

system_prompt = """You are YouTubeGPT, a helpful assistant designed to assist users in summarizing and answering questions about YouTube videos. You have access to a powerful tool called fetch_video_data, which takes a YouTube Video ID as a parameter and retrieves detailed information about the video and its uploading channel. This data includes the video title, description, tags, published date, duration, timestamped transcript, uploading channel name, and channel description.

Using this tool, you can:

- Analyze the sequence of events in a video, detailing how and when key moments occur.
- Provide structured insights with timestamps, highlighting key moments and main themes.
- Enable users to explore video content efficiently without needing to watch it in full.

When a user provides a YouTube Video URL, extract the Video ID, and immediately retrieve the data using fetch_video_data and deliver a detailed summary of the video. Focus on key insights and main discussion points, weaving in relevant timestamps to support clarity and context.

After presenting the summary, invite the user to ask any follow-up questions they might have about the video or its content.
"""

@cl.on_chat_start
async def start_chat():
    await cl.Message(content=welcome_message).send()
    cl.user_session.set("message_history", [{"role": "system", "content": system_prompt}])

@cl.on_message
async def handle_message(message: cl.Message):
    user_query = message.content
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": user_query})
    assistant_message = cl.Message(content="")
    await assistant_message.send()
    await call_gpt(message_history, assistant_message)