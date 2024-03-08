from typing import List

from .message import Message
from .base_message_history_repository import BaseMessageHistoryRepository


class MessageHistory:
    def __init__(self, session_id: str, repository: BaseMessageHistoryRepository):
        self.repository = repository
        self.session_id = session_id

    def add_ai_message(self, message):
        self.add_message('AI', message)

    def add_user_message(self, message):
        self.add_message('User', message)

    def add_system_message(self, message):
        self.add_message('System', message)

    def add_message(self, role: str, content: str) -> None:
        message = Message(self.session_id, role, content)
        self.repository.save_message(message)

    def get_history(self) -> List[dict]:
        return [message.dict() for message in self.repository.get_all_messages(self.session_id)]
