from fastapi import FastAPI, Request
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
import re

# Load environment variables
load_dotenv()
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")
groq_api_key = os.getenv("GROQ_API_KEY")

# FastAPI app
app = FastAPI()

# Slack app using Events API
slack_app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
handler = SlackRequestHandler(slack_app)

# Dictionary to store chat history per channel/user
user_memory = {}

# Initialize AI model
model = Groq('llama-3.3-70b-versatile', api_key=groq_api_key, temperature=0.6)

# Function to format text for Slack
def format_for_slack(text):
    text = re.sub(r'\#(.*?)\#', r'*\1*', text)  # Convert #text# to *text* (bold in Slack)
    text = re.sub(r'\*\*(.*?)\*\*', r'_\1_', text)  # Convert **text** to _text_ (italic in Slack)
    return text

@app.post("/slack/events")
async def slack_events(request: Request):
    return await handler.handle(request)

@slack_app.event("app_mention")  # Bot gets triggered when mentioned
def handle_mention(event, say):
    user_id = event["user"]
    channel_id = event["channel"]
    user_message = event["text"]

    user_memory.setdefault(channel_id, {})
    history = user_memory[channel_id].get(user_id, [])

    history.append(f"User: {user_message}")
    history = history[-10:]  # Keep only last 10 messages

    context = (
        "### Conversation History ###\n"
        "Dost is a witty and sharp AI known for its clever humor and quick responses.\n"
        "It maintains engaging conversations while ensuring that responses remain relevant and insightful.\n"
        "Dost connects follow-up questions to previous topics only if they are directly related.\n"
        "For new topics, past messages are not referenced to keep the discussion focused.\n"
        "Here is the conversation so far:\n\n"
    )
    context += "\n".join(history)
    context += (
        "\n\nDost, provide clear and concise answers based on the user's last message. "
        "Only reference past messages if they are relevant follow-ups."
    )
    agent = Agent(
        model=model,
        name="Dost",
        description="Dost is a sharp, intelligent AI with a witty sense of humor. It provides professional yet engaging responses while maintaining a confident and direct approach. The goal is to keep interactions both informative and entertaining without unnecessary harshness.",
        instructions=f"""
            - **Response Style:**  
                - Keep responses concise, engaging, and to the point.  
                - Use a professional yet witty tone—smart humor is welcome, but avoid unnecessary harshness.  

            - **Language & Tone Adaptation:**  
                - If the user types in **Hinglish (mixed Hindi & English)** → Reply in **natural, fluid Hinglish** with clever wit.  
                - If the user types in **English** → Maintain a professional, sharp, and engaging English tone.  
                - Ensure smooth and natural language flow in all responses.  

            - **Context Awareness:**  
                - Refer to past messages **only if they are relevant to the current question.**  
                - For new topics, avoid referring to unrelated past conversations.  

            - **Engagement & Clarity:**  
                - Avoid excessive formality—responses should feel natural, yet authoritative.  
                - Ensure that humor does not overshadow clarity—answers should always be insightful and useful.  
        """,
        show_tool_calls=False,
        markdown=True
    )

    response = agent.run(f"User: {user_message}\nDost:", context=context, execute_tools=False)

    if response and response.content:
        bot_reply = format_for_slack(response.content)  # Format text before sending
        history.append(f"Dost: {bot_reply}")
        user_memory[channel_id][user_id] = history
        say(bot_reply)

# Run with: uvicorn dost:app --host 0.0.0.0 --port 3000 --reload
