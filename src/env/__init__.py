from typing import Any

class Env:
    def reset(self: "Env") -> None:
        raise NotImplementedError()

    def step(self: "Env", action: Any, *args, **kwargs) -> int:
        raise NotImplementedError()
