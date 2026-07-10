"""
Handle user interrupting assistant mid-response. 
Detects VAD activity during TTS playback, stops TTS, captures new input, 
and decides whether to incorporate or override the current response.
"""

class InterruptionHandler:
    def __init__(self):
        raise NotImplementedError("This module is a stub for future development.")
