import re

def clean_text(text):
    return re.sub(r'[^\x20-\x7E]', '', text)

def generate_prompt_template(json_data):
    if not isinstance(json_data, dict):
        return "Error: Invalid JSON format. Expected a dictionary."

    video_title = clean_text(json_data.get('video_name', 'N/A'))
    video_description = clean_text(json_data.get('video_description', 'N/A'))
    video_tags = [clean_text(tag) for tag in json_data.get('video_tags', [])]
    video_published_date = clean_text(json_data.get('video_published_date', 'N/A'))
    video_duration = clean_text(json_data.get('video_duration', 'N/A'))
    video_transcript = clean_text(json_data.get('video_transcript', 'N/A'))
    channel_name = clean_text(json_data.get('channel_name', 'N/A'))
    channel_description = clean_text(json_data.get('channel_description', 'N/A'))

    prompt_template = f"""You are an advanced AI specializing in summarizing YouTube video metadata and transcripts.

### Video Summary  
#### Channel: {channel_name}  
Channel Description: {channel_description}  

#### Video: {video_title} | {video_published_date}  
Video Description: {video_description}  
Video Duration: {video_duration}  
Video Tags: {', '.join(video_tags)}  

#### Transcript Summary:  
{video_transcript}  

"Provide a detailed summary of this video, emphasizing key insights and main discussion points. Incorporate supporting timestamps throughout to enhance clarity.

Once the summary is complete, encourage the user to ask any follow-up questions they may have.
"""

    return prompt_template