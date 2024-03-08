from abc import ABC, abstractmethod
from typing import List

from .message import Message


class BaseMessageHistoryRepository(ABC):
    @abstractmethod
    def save_message(self, message: Message) -> None:
        pass

    @abstractmethod
    def get_all_messages(self, session_id: str) -> List[Message]:
        pass

    @abstractmethod
    def clear_old_messages(self, session_id: str) -> None:
        pass
