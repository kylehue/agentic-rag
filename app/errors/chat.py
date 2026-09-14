class ChatNotFoundError(Exception):
    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        super().__init__(f"Chat '{chat_id}' not found.")


class ChatForbiddenError(Exception):
    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        super().__init__(f"Chat '{chat_id}' belongs to another user.")
