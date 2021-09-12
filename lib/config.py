import os

from dotenv import load_dotenv

load_dotenv()

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")
AUTHOR = os.getenv("AUTHOR")
