from enum import Enum


class Connection(Enum):
    DISCONNECTED = 'disconnected'
    SUCCESS = 'success'
    NO_MATCH = 'no match'
    TRY_AGAIN_LATER = 'Try Again later'
