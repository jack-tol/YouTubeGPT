# YouTubeGPT

Welcome to the GitHub repository for **YouTubeGPT**, an AI-powered tool that allows users to interact with YouTube videos through dynamic conversations. By leveraging Long Context Language Models (LCLMs), YouTubeGPT enables users to ask questions and gain structured insights about a video’s content without having to watch the entire thing.

## About

YouTubeGPT extracts and processes key video information—including the title, transcript, and metadata—to provide detailed, context-aware responses to user queries. This allows for a deeper understanding of the sequence of events, main themes, and critical moments within the video. 

If you're interested in about the motivation & rationale of YouTubeGPT, check out the [blog post](https://jacktol.net/posts/introducing_youtubegpt/).

## How does it work?

YouTubeGPT simplifies video analysis by turning YouTube videos into an interactive text-based experience. Instead of manually searching through a video, users can ask direct questions and receive detailed, AI-generated responses.

1. **Enter a YouTube URL**: Paste the link to a YouTube video into the chat box.
2. **Processing & Analysis**: The system extracts and processes the video transcript, title, and metadata, providng you with an initial video summary.
3. **Ask Your Questions**: Query the video about specific moments, concepts, or topics covered in the content.
4. **Receive Answers**: YouTubeGPT provides clear and structured responses, helping users navigate the video efficiently.

### Features

- **Understand Video Content**: Ask about the main ideas, arguments, and conclusions of a video.
- **Event Sequence Analysis**: Gain insights into how things happen and in what order.
- **Detailed Summaries**: Get structured overviews, breaking down key points from the video.
- **Metadata Extraction**: Retrieve information such as the video title, description, duration, and channel details.
- **Efficient Navigation**: Save time by skipping irrelevant sections and focusing on what matters.

### Instructions

1. **Paste a YouTube URL** into the YouTubeGPT interface.
2. **Wait for Processing**: The system will extract the transcript and metadata.
3. **Ask Questions**:
   - Get an overview of the video’s main topics.
   - Request insights into specific segments.
   - Clarify details about the content.
4. **Follow-Up & Exploration**:
   - Ask additional questions to refine your understanding.
   - Explore related concepts based on the video’s content.

## YouTubeGPT in Action

YouTubeGPT is designed for students, researchers, and content consumers who want a faster, more interactive way to engage with video material. Whether you need a high-level summary or an in-depth breakdown, this tool streamlines the process of consuming YouTube content.

## YouTube Scraping Tool Usage

After installing the requirements, simply run the `example_usage.py` script. This script imports the tool and runs data scraping on a YouTube video. The function takes the YouTube video ID as an argument and returns a JSON object containing all of the scraped data for further use. The script then prints the result to demonstrate the format and scope of the retrieved data.

## Run It Yourself

After installing the requirements, simply add your `OPENAI_API_KEY` to a `.env` file, then run `chainlit run app.py` in your terminal. The app will open automatically in your web browser, allowing you to submit YouTube video URLs along with your questions in the chat input box and start receiving summaries and answers instantly.

For any inquiries, feedback, or suggestions, reach out via [jacktol.net](https://jacktol.net/).