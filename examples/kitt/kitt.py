# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import time
from livekit.plugins.elevenlabs import (TTS, Voice, VoiceSettings)
from livekit.plugins.deepgram import STT
from chatgpt import (
    ChatGPTMessage,
    ChatGPTMessageRole,
    ChatGPTPlugin,
)
from livekit.agents.tts import SynthesisEvent, SynthesisEventType
from livekit import rtc, agents, api
from typing import AsyncIterable
import logging
import json
from enum import Enum
from datetime import datetime
import asyncio
import os
from dotenv import load_dotenv
abs_dir = os.path.dirname(os.path.abspath(__file__))

dotenv_path = os.path.abspath(os.path.join(abs_dir, '../..', '.env'))
load_dotenv(dotenv_path)

prompt_path = os.path.join(abs_dir, 'prompt.txt')
promptFile = open(prompt_path, "r")

PROMPT = promptFile.read()
INTRO = "Hello, thanks for joining. Let me know when you're ready to get started."
# SIP_INTRO = "Hello, I am KITT, a friendly voice assistant powered by LiveKit Agents. \
#              Feel free to ask me anything — I'm here to help! Just start talking."


# convert intro response to a stream
async def intro_text_stream():
    yield INTRO


AgentState = Enum("AgentState", "IDLE, LISTENING, THINKING, SPEAKING")

STT_SILENCE_BUFFER = 1000

ELEVEN_TTS_SAMPLE_RATE = 24000
ELEVEN_TTS_CHANNELS = 1

ELEVEN_CHARLIE = Voice(
    id="IKne3meq5aSn9XLyUdCD",
    name="Charlie",
    category="premade",
    # settings=VoiceSettings(
    #     stability=0.55, similarity_boost=0.75, style=0.0, use_speaker_boost=True
    # ),
)

# v2 only
ELEVEN_RANDALL = Voice(
    id="sGmoTXtWfLdnlMyLWWgH",
    name="Randall",
    category="cloned",
    # settings=VoiceSettings(
    #     stability=0.55, similarity_boost=0.75, style=0.0, use_speaker_boost=True
    # ),
)

# more friendly and youthful. maybe a little too "posh"
ELEVEN_BELINDA = Voice(
    id="ibcyNEUbOlXZVwBfzZdH",
    name="Belinda",
    category="generated",
)

# Fast & clipped
ELEVEN_AMELIA = Voice(
    id="jIsuAyA6m00dz2fj15q0",
    name="Amelia",
    category="generated",
)

use_voice = ELEVEN_BELINDA

ELEVEN_MODEL_ID = "eleven_multilingual_v1"


class KITT:
    @classmethod
    async def create(cls, ctx: agents.JobContext):
        kitt = KITT(ctx)
        await kitt.start()

    def __init__(self, ctx: agents.JobContext):
        # plugins
        self.chatgpt_plugin = ChatGPTPlugin(
            prompt=PROMPT, message_capacity=20, model="gpt-4-1106-preview"
        )
        self.stt_plugin = STT(
            language="en",
            interim_results=True,
            min_silence_duration=STT_SILENCE_BUFFER,
        )
        self.tts_plugin = TTS(
            model_id=ELEVEN_MODEL_ID, sample_rate=ELEVEN_TTS_SAMPLE_RATE, voice=use_voice
        )

        self.ctx: agents.JobContext = ctx
        self.chat = rtc.ChatManager(ctx.room)
        self.audio_out = rtc.AudioSource(
            ELEVEN_TTS_SAMPLE_RATE, ELEVEN_TTS_CHANNELS)

        self._sending_audio = False
        self._processing = False
        self._agent_state: AgentState = AgentState.IDLE

        self.user_connected: float = 0
        self.user_audio_muted = False
        self.unsent_messages = []

        self.chat.on("message_received", self.on_chat_received)
        self.ctx.room.on("track_subscribed", self.on_track_subscribed)
        self.ctx.room.on("track_muted", self.on_track_muted)
        self.ctx.room.on("track_unmuted", self.on_track_unmuted)

        self.ctx.room.on("participant_connected", self.on_participant_connected)
        self.ctx.room.on("participant_disconnected", self.on_participant_disconnected)

    async def start(self):
        # if you have to perform teardown cleanup, you can listen to the disconnected event
        # self.ctx.room.on("disconnected", your_cleanup_function)

        # publish audio track
        track = rtc.LocalAudioTrack.create_audio_track(
            "agent-mic", self.audio_out)
        await self.ctx.room.local_participant.publish_track(track)

        print("🤖 KITT started", self.ctx.room.name)

        await asyncio.sleep(1)
        await self.send_intro_message()

    async def send_intro_message(self):
        # check if the participant is ready
        if (self.user_connected and (self.user_connected + 3 < time.time())):
            await self.process_chatgpt_result(intro_text_stream())
            self.update_state()
        else:
            # give the participant a second to get properly connected
            await asyncio.sleep(1)
            await self.send_intro_message()

    def on_chat_received(self, message: rtc.ChatMessage):
        # TODO: handle deleted and updated messages in message context
        if message.deleted:
            return

        msg = ChatGPTMessage(role=ChatGPTMessageRole.user,
                             content=message.message)
        chatgpt_result = self.chatgpt_plugin.add_message(msg)
        self.ctx.create_task(self.process_chatgpt_result(chatgpt_result))

    def on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        self.ctx.create_task(self.process_track(track))

    def on_track_muted(
        self,
        participant: rtc.RemoteParticipant,
        publication: rtc.TrackPublication,
    ):
        if (publication.track.kind != rtc.TrackKind.KIND_AUDIO or participant.identity == "kitt_agent"):
            return

        self.user_audio_muted = publication.track.muted

        if publication.track.muted:
            # Wait for a bit before sending unsent messages to make sure it's been captured after the mute event
            # time.sleep(2)
            self.post_unsent_messages()

    def on_track_unmuted(
        self,
        participant: rtc.RemoteParticipant,
        publication: rtc.TrackPublication,
    ):
        if (publication.track.kind != rtc.TrackKind.KIND_AUDIO or participant.identity == "kitt_agent"):
            return

        self.user_audio_muted = publication.track.muted

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        print("👋 Participant connected", participant.identity)
        if (not participant.identity == "kitt_agent"):
            # set the time that the user connected so that the intro message can be sent a few seconds later
            self.user_connected = time.time()

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        print("👋 Participant disconnected", participant.identity)
        # if (not participant.identity == "kitt_agent"):
        #     asyncio.run(self.close_room)

    async def close_room(self):
        room_info = await api.DeleteRoomRequest(room_sid=self.ctx.room.sid)
        print("👋 Room closed", room_info)

    async def process_track(self, track: rtc.Track):
        audio_stream = rtc.AudioStream(track)
        stream = self.stt_plugin.stream()
        self.ctx.create_task(self.process_stt_stream(stream))
        async for audio_frame in audio_stream:
            if self._agent_state != AgentState.LISTENING:
                continue
            stream.push_frame(audio_frame)
        await stream.flush()

    async def process_stt_stream(self, stream):
        buffered_text = ""
        async for event in stream:
            print(
                "💬 ", 
                'is_final=' + ('✅' if event.is_final else '☒'),
                'end_of_speech=' + ('✅' if event.end_of_speech else '☒'),
                'buffered_text= ' + buffered_text + " + " + event.alternatives[0].text,
                'unsent_messages= ' + "; ".join(self.unsent_messages),
            )
            if event.alternatives[0].text == "":
                continue
            if event.is_final:
                buffered_text = " ".join([buffered_text, event.alternatives[0].text])

            if not event.end_of_speech:
                continue
            if event.is_final:
                buffered_text = " ".join([buffered_text, event.alternatives[0].text])

            self.unsent_messages.append(buffered_text)

            await self.ctx.room.local_participant.publish_data(
                json.dumps(
                    {
                        "text": buffered_text,
                        "timestamp": int(datetime.now().timestamp() * 1000),
                    }
                ),
                topic="transcription",
            )

            # msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=buffered_text)
            # chatgpt_stream = self.chatgpt_plugin.add_message(msg)
            # self.ctx.create_task(self.process_chatgpt_result(chatgpt_stream))

            buffered_text = ""

            if self.user_audio_muted:
                self.post_unsent_messages()

    def post_unsent_messages(self):
        print("💥 Posting unsent messages", self.unsent_messages)
        if len(self.unsent_messages) > 0:
            text = "\n".join(self.unsent_messages)
            self.unsent_messages = []
            msg = ChatGPTMessage(role=ChatGPTMessageRole.user, content=text)
            chatgpt_stream = self.chatgpt_plugin.add_message(msg)
            self.ctx.create_task(self.process_chatgpt_result(chatgpt_stream))
            buffered_text = ""

    async def process_chatgpt_result(self, text_stream):
        print("🧠 Processing ChatGPT result", text_stream)
        # ChatGPT is streamed, so we'll flip the state immediately
        self.update_state(processing=True)

        stream = self.tts_plugin.stream()
        # send audio to TTS in parallel
        self.ctx.create_task(self.send_audio_stream(stream))
        all_text = ""
        async for text in text_stream:
            stream.push_text(text)
            all_text += text

        self.update_state(processing=False)
        # buffer up the entire response from ChatGPT before sending a chat message
        await self.chat.send_message(all_text)
        await stream.flush()

    async def send_audio_stream(self, tts_stream: AsyncIterable[SynthesisEvent]):
        async for e in tts_stream:
            if e.type == SynthesisEventType.STARTED:
                self.update_state(sending_audio=True)
            elif e.type == SynthesisEventType.FINISHED:
                self.update_state(sending_audio=False)
            elif e.type == SynthesisEventType.AUDIO:
                await self.audio_out.capture_frame(e.audio.data)
        await tts_stream.aclose()

    def update_state(self, sending_audio: bool = None, processing: bool = None):
        if sending_audio is not None:
            self._sending_audio = sending_audio
        if processing is not None:
            self._processing = processing

        state = AgentState.LISTENING
        if self._sending_audio:
            state = AgentState.SPEAKING
        elif self._processing:
            state = AgentState.THINKING

        self._agent_state = state
        metadata = json.dumps(
            {
                "agent_state": state.name.lower(),
            }
        )
        self.ctx.create_task(
            self.ctx.room.local_participant.update_metadata(metadata))


if __name__ == "__main__":
    # logging.basicConfig(level=logging.INFO)
    logging.basicConfig(level=logging.WARNING)

    async def job_request_cb(job_request: agents.JobRequest):
        logging.info("Accepting job for KITT")

        await job_request.accept(
            KITT.create,
            identity="kitt_agent",
            name="KITT",
            auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY,
            auto_disconnect=agents.AutoDisconnect.DEFAULT,
        )

    worker = agents.Worker(request_handler=job_request_cb)
    agents.run_app(worker)
