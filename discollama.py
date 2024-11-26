import io
import os
import json
import asyncio
import argparse
from datetime import datetime, timedelta

import ollama
import discord
from discord.ext import commands
from discord import app_commands
import redis

from logging import getLogger

# piggy back on the logger discord.py set up
logging = getLogger('discord.discollama')


class Response:
    def __init__(self, message):
        self.message = message
        self.channel = message.channel

        self.r = None
        self.sb = io.StringIO()

    async def write(self, s, end=''):
        if self.sb.seek(0, io.SEEK_END) + len(s) + len(end) > 2000:
            self.r = None
            self.sb.seek(0, io.SEEK_SET)
            self.sb.truncate()

        self.sb.write(s)

        value = self.sb.getvalue().strip()
        if not value:
            return

        if self.r:
            await self.r.edit(content=value + end)
            return

        self.r = await self.channel.send(value)


class Discollama:
    def __init__(self, ollama, redis, model):
        self.ollama = ollama
        self.redis = redis
        self.model = model
        self.spam_mode = {}  # Dictionary to toggle spam mode per channel

        # Initialize command bot
        intents = discord.Intents.default()
        intents.message_content = True
        self.discord = commands.Bot(command_prefix="!", intents=intents)
        self.tree = self.discord.tree

        # Register event handlers
        self.discord.event(self.on_ready)
        self.discord.event(self.on_message)

        self.setup_slash_commands()

    async def on_ready(self):
        activity = discord.Activity(name='AI Sandbox Bot', state='Ask me anything!', type=discord.ActivityType.custom)
        await self.discord.change_presence(activity=activity)
        logging.info(
            'running aisandbox bot 3.7.2 on docker-py',
        )

        logging.info(
            'Ready! Invite URL: %s',
            discord.utils.oauth_url(
                self.discord.application_id,
                permissions=discord.Permissions(
                    administrator=True
                ),
                scopes=['bot'],
            ),
        )
        await self.tree.sync()

    async def on_message(self, message):
        if self.discord.user == message.author:
            return

        if not self.spam_mode.get(message.channel.id, False) and not self.discord.user.mentioned_in(message):
            return

        content = message.content.replace(f'<@{self.discord.user.id}>', '').strip()
        if not content:
            content = f'{message.author.display_name}: Hi!'
        else:
            content = f'{message.author.display_name}: {content}'

        channel = message.channel
        messages = []
        async for msg in channel.history(limit=51):
            messages.append(f"{msg.author.display_name}: {msg.content}")

        recent_messages = messages[1:]
        past_messages = '\n'.join(recent_messages)[:2048]

        context = f"Recent Messages. Use them as context. These are your memory:\n{past_messages}\nMessage to respond to:\n{content}"
        logging.info("Submitting prompt to model: %s", context)
        context_list = [int(x) for x in context.split() if x.isdigit()]

        r = Response(message)
        task = asyncio.create_task(self.thinking(message))
        async for part in self.generate(context):
            task.cancel()

            await r.write(part['response'], end='...')

        await r.write('')
        await self.save(r.channel.id, message.id, part['context'])

    async def thinking(self, message, timeout=999):
        try:
            await message.add_reaction('🤔')
            async with message.channel.typing():
                await asyncio.sleep(timeout)
        except Exception:
            pass
        finally:
            await message.remove_reaction('🤔', self.discord.user)

    async def generate(self, content):
        sb = io.StringIO()

        t = datetime.now()
        async for part in await self.ollama.generate(model=self.model, prompt=content, keep_alive=-1, stream=True):
            sb.write(part['response'])

            if part['done'] or datetime.now() - t > timedelta(seconds=1):
                part['response'] = sb.getvalue()
                yield part
                t = datetime.now()
                sb.seek(0, io.SEEK_SET)
                sb.truncate()

    async def save(self, channel_id, message_id, ctx: list[int]):
        logging.info('Received message')

    async def load(self, channel_id=None, message_id=None) -> list[int]:
        if channel_id:
            message_id = self.redis.get(f'discollama:channel:{channel_id}')

        ctx = self.redis.get(f'discollama:message:{message_id}')
        return json.loads(ctx) if ctx else []

    def setup_slash_commands(self):
        @self.tree.command(name="toggle-yap", description="Make it respond to every message in this channel.")
        async def toggle_yap(interaction: discord.Interaction):
            channel_id = interaction.channel_id
            self.spam_mode[channel_id] = not self.spam_mode.get(channel_id, False)
            await interaction.response.send_message(f"Yap mode {'enabled' if self.spam_mode[channel_id] else 'disabled'} for this channel.")

    def run(self, token):
        try:
            self.discord.run(token)
        except Exception:
            self.redis.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ollama-model', default=os.getenv('OLLAMA_MODEL', 'mymodel'), type=str)

    parser.add_argument('--redis-host', default=os.getenv('REDIS_HOST', '127.0.0.1'), type=str)
    parser.add_argument('--redis-port', default=os.getenv('REDIS_PORT', 6379), type=int)

    parser.add_argument('--buffer-size', default=32, type=int)

    args = parser.parse_args()

    Discollama(
        ollama.AsyncClient(),
        redis.Redis(host=args.redis_host, port=args.redis_port, db=0, decode_responses=True),
        model=args.ollama_model,
    ).run(os.environ['DISCORD_TOKEN'])


if __name__ == '__main__':
    main()
