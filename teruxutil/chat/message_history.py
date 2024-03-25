from typing import List

from .message import Message
from .base_message_history_repository import BaseMessageHistoryRepository

ROLE_AI = 'AI'
ROLE_USER = 'User'
ROLE_SYSTEM = 'System'

class MessageHistory:
    def __init__(self, *, session_id: str, repository: BaseMessageHistoryRepository):
        self.repository = repository
        self.session_id = session_id

    def add_ai_message(self, message: str):
        self.add_message(ROLE_AI, message)

    def add_user_message(self, message: str):
        self.add_message(ROLE_USER, message)

    def add_system_message(self, message: str):
        self.add_message(ROLE_SYSTEM, message)

    def add_message(self, role: str, message: str) -> None:
        message_obj = Message(session_id=self.session_id, role=role, content=message)
        self.repository.save_message(message_obj)
        self.repository.clear_old_messages(self.session_id)

    def get_history(self) -> List[Message]:
        return [message for message in self.repository.get_all_messages(self.session_id)]
