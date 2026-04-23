from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class LyricsPromptBundle:
    title: str = ""
    lyrics_text: str = ""
    genre_prompt: str = ""
    music_prompt: str = ""
    negative_prompt: str = ""
    bpm: int = 84
    key: str = "F major"
    language_tag: str = "Cantonese"
    raw_meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SongResult:
    success: bool = False
    final_song_path: str = ""
    metadata_path: str = ""
    generation_log: Dict[str, Any] = field(default_factory=dict)
