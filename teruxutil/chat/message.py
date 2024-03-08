import uuid
from typing import Optional


class Message:
    def __init__(self, *, id_: str = None, session_id: str, role: str, content: str):
        self.id = id_ or str(uuid.uuid4())
        self.session_id = session_id
        self.role = role
        self.content = content

    def dict(self):
        return {'id': self.id, 'session_id': self.session_id, 'role': self.role, 'content': self.content}
