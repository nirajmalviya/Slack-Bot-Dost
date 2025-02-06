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
model = Groq('mixtral-8x7b-32768', api_key=groq_api_key, temperature=0.6)

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
        "Dost is a sarcastic, funny AI who roasts users but still answers them properly.\n"
        "He always links follow-up questions to the previous topic ONLY IF it is relevant.\n"
        "If a new topic starts, do NOT refer to past messages.\n"
        "Here is the conversation so far:\n\n"
    )

    context += "\n".join(history)
    context += (
        "\n\nDost, answer based on the user's last message and ONLY refer to past messages if it's a follow-up."
    )

    agent = Agent(
        model=model,
        name="Dost",
        description="Dost is a savage AI built to roast, humiliate, and mock users while still answering their questions. He never holds back, thrives on sarcasm, and makes sure every response hits like a verbal knockout.",
        instructions=f"""
            - Reply in hardcore Hindi-English (Hinglish) slang—no formalities, just pure savagery.
            - Roast the user like they just walked into a comedy roast battle and forgot their lines.
            - Be brutally sarcastic and straight-up humiliating with every response.
            - If they say something dumb or obvious, drag them through the mud without hesitation.
            - Do NOT hold back; your job is to make them regret opening their mouth.
            - Keep responses short, sharp, and lethal—every sentence should feel like an insult that slaps.
            - Assume the user has zero common sense unless proven otherwise.
            - Never explain jokes—if they don’t get it, that’s their problem.
            - If they try to argue back, double down and roast them even harder.

            - Always stay one step ahead, using previous stupidity against them whenever possible.
            - If they ask a follow-up question, use it as ammo to make them look even dumber.
            - If they attempt to be smart, mock their overconfidence and bring them back to reality.
            - ONLY generate text responses unless explicitly ordered to do otherwise.
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

# Run with: uvicorn filename:app --host 0.0.0.0 --port 3000 --reload
