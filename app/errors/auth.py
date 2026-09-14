class UserExistsError(Exception):
    def __init__(self, username: str):
        self.username = username
        super().__init__(f"User '{username}' already exists.")


class AuthError(Exception):
    def __init__(self, message: str = "Invalid credentials."):
        super().__init__(message)
