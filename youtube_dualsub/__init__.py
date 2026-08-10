"""youtube_dualsub — local bilingual subtitles for YouTube.

Pipeline (strictly serial, each stage releases its VRAM before the next starts):

    audio -> vocals -> asr -> sentences -> context -> translate -> shape

The serial design is what keeps peak VRAM under ~9 GB even though the stages
together would need far more than the 14 GB available.
"""

__version__ = "0.1.0"
