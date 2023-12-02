from pydantic import BaseModel


class Color(BaseModel):
    name: str

    def get_ansi_color_code(self: 'Color') -> str:
        return {
            "red": "\033[31m",
            "green": "\033[32m",
            "blue": "\033[34m",
            "yellow": "\033[33m",
        }.get(self.name, "\033[0m")
