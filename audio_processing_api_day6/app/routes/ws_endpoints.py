"""
Real-time WebSocket endpoints for streaming audio processing.
Supports live transcription and real-time voice communication.
"""
import asyncio
import json
import logging
import time
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.asr_model import ASRModel
from app.utils.audio_processing import AudioProcessor

router = APIRouter()
logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections and real-time processing."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.asr_model = ASRModel()
        self.audio_processor = AudioProcessor()

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        """Remove client connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: str, client_id: str):
        """Send message to specific client."""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_text(message)

    async def broadcast(self, message: str):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws/realtime-transcription/{client_id}")
async def websocket_transcription(websocket: WebSocket, client_id: str):
    """
    Real-time audio transcription via WebSocket.
    Supports streaming audio input with live transcription output.
    Includes confidence scoring and partial results.
    """
    await manager.connect(websocket, client_id)

    try:
        # Send initial connection confirmation
        await manager.send_personal_message(
            json.dumps({
                "type": "connection_established",
                "client_id": client_id,
                "supported_formats": ["wav", "mp3", "webm"],
                "sample_rate": 16000
            }),
            client_id
        )

        audio_buffer = bytearray()
        processing_active = False

        while True:
            data = await websocket.receive()

            if data["type"] == "websocket.receive":
                if "bytes" in data:
                    # Handle binary audio data
                    audio_chunk = data["bytes"]
                    audio_buffer.extend(audio_chunk)

                    # Process when buffer reaches threshold (e.g., 1 second of audio)
                    if len(audio_buffer) >= 32000 and not processing_active:  # ~1 sec at 16kHz
                        processing_active = True
                        try:
                            chunk_to_process = bytes(audio_buffer[:32000])
                            audio_buffer = audio_buffer[32000:]

                            result = await manager.asr_model.transcribe(chunk_to_process)

                            response = {
                                "type": "transcription_result",
                                "text": result["text"],
                                "language": result["language"],
                                "confidence": result["language_probability"],
                                "is_partial": len(audio_buffer) > 0,
                                "timestamp": time.time()
                            }

                            await manager.send_personal_message(
                                json.dumps(response),
                                client_id
                            )

                        except Exception as e:
                            error_response = {
                                "type": "error",
                                "message": f"Processing error: {str(e)}",
                                "timestamp": time.time()
                            }
                            await manager.send_personal_message(
                                json.dumps(error_response),
                                client_id
                            )
                        finally:
                            processing_active = False

                elif "text" in data:
                    # Handle text commands
                    try:
                        command = json.loads(data["text"])
                        await handle_ws_command(command, client_id)
                    except json.JSONDecodeError:
                        await manager.send_personal_message(
                            json.dumps({"type": "error", "message": "Invalid JSON command"}),
                            client_id
                        )

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        manager.disconnect(client_id)

async def handle_ws_command(command: Dict[str, Any], client_id: str):
    """Handle WebSocket text commands."""
    command_type = command.get("type")

    if command_type == "ping":
        response = {
            "type": "pong",
            "timestamp": time.time()
        }
        await manager.send_personal_message(json.dumps(response), client_id)

    elif command_type == "set_language":
        language = command.get("language", "auto")
        response = {
            "type": "language_updated",
            "language": language,
            "timestamp": time.time()
        }
        await manager.send_personal_message(json.dumps(response), client_id)

    elif command_type == "get_status":
        status = {
            "type": "status",
            "client_id": client_id,
            "connected": True,
            "active_connections": len(manager.active_connections),
            "timestamp": time.time()
        }
        await manager.send_personal_message(json.dumps(status), client_id)

    else:
        error_response = {
            "type": "error",
            "message": f"Unknown command type: {command_type}",
            "timestamp": time.time()
        }
        await manager.send_personal_message(json.dumps(error_response), client_id)

@router.websocket("/ws/voice-chat/{room_id}")
async def websocket_voice_chat(websocket: WebSocket, room_id: str):
    """
    Real-time voice chat with live transcription and translation.
    Supports multi-user rooms with automatic language detection.
    """
    client_id = f"{room_id}_{id(websocket)}"
    await manager.connect(websocket, client_id)

    try:
        # Send room join confirmation
        join_message = {
            "type": "room_joined",
            "room_id": room_id,
            "client_id": client_id,
            "participants": len([c for c in manager.active_connections.keys() if room_id in c])
        }
        await manager.send_personal_message(json.dumps(join_message), client_id)

        # Notify other room participants
        participant_message = {
            "type": "participant_joined",
            "room_id": room_id,
            "new_participant": client_id
        }

        for conn_id, connection in manager.active_connections.items():
            if room_id in conn_id and conn_id != client_id:
                await connection.send_text(json.dumps(participant_message))

        while True:
            data = await websocket.receive()

            if data["type"] == "websocket.receive":
                if "bytes" in data:
                    audio_data = data["bytes"]
                    try:
                        result = await manager.asr_model.transcribe(audio_data)
                        chat_message = {
                            "type": "voice_message",
                            "room_id": room_id,
                            "sender": client_id,
                            "text": result["text"],
                            "language": result["language"],
                            "timestamp": time.time(),
                            "audio_duration": result.get("duration", 0)
                        }
                        for conn_id, connection in manager.active_connections.items():
                            if room_id in conn_id:
                                await connection.send_text(json.dumps(chat_message))
                    except Exception as e:
                        error_message = {
                            "type": "processing_error",
                            "message": str(e),
                            "timestamp": time.time()
                        }
                        await manager.send_personal_message(json.dumps(error_message), client_id)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        leave_message = {
            "type": "participant_left",
            "room_id": room_id,
            "departed_participant": client_id
        }
        for conn_id, connection in manager.active_connections.items():
            if room_id in conn_id and conn_id != client_id:
                try:
                    await connection.send_text(json.dumps(leave_message))
                except Exception:
                    pass  # Connection might be closed
