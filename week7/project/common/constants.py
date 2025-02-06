from enum import Enum, auto


class FileType(Enum):
    TRANSCRIPT = auto()
    SUMMARY = auto()
    AUDIO = auto()


class FileExtension(Enum):
    JSONL = ".jsonl"
    TXT = ".txt"


class MimeType(Enum):
    AUDIO = "audio/wav"
