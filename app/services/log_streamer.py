import asyncio
from typing import Dict, List
import time

class LogStreamer:
    """
    In-memory PubSub for streaming installation and build logs to the frontend.
    """
    def __init__(self):
        # Maps workflow_id to a list of queues (one queue per active client connection)
        self._queues: Dict[int, List[asyncio.Queue]] = {}
        # Buffer to keep the last N lines so new connections can catch up
        self._buffers: Dict[int, List[dict]] = {}
        self._max_buffer_size = 1000

    def publish(self, workflow_id: int, message: str, phase: str = "dependency"):
        if workflow_id not in self._buffers:
            self._buffers[workflow_id] = []
        if workflow_id not in self._queues:
            self._queues[workflow_id] = []

        log_entry = {
            "timestamp": time.time(),
            "phase": phase,
            "message": message
        }

        self._buffers[workflow_id].append(log_entry)
        if len(self._buffers[workflow_id]) > self._max_buffer_size:
            self._buffers[workflow_id].pop(0)

        # Non-blocking put to all connected clients
        for q in self._queues[workflow_id]:
            try:
                q.put_nowait(log_entry)
            except asyncio.QueueFull:
                pass # Drop if client is too slow

    async def subscribe(self, workflow_id: int):
        q = asyncio.Queue(maxsize=1000)
        
        if workflow_id not in self._queues:
            self._queues[workflow_id] = []
        self._queues[workflow_id].append(q)

        # Yield historical buffered logs first
        if workflow_id in self._buffers:
            for log_entry in self._buffers[workflow_id]:
                yield log_entry

        try:
            while True:
                # Wait for new logs
                log_entry = await q.get()
                yield log_entry
        finally:
            self._queues[workflow_id].remove(q)
            if not self._queues[workflow_id]:
                del self._queues[workflow_id]

    def get_logs(self, workflow_id: int) -> List[dict]:
        """Return all currently buffered logs for polling fallback."""
        return self._buffers.get(workflow_id, [])

    def clear(self, workflow_id: int):
        self._buffers.pop(workflow_id, None)

# Global singleton
log_streamer = LogStreamer()
