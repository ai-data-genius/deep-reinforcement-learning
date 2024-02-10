from os import makedirs
from os.path import exists, join
from typing import Any

from torch import save
from torch.nn import Module


class Network(Module):
    def init_network(self: "Network") -> None:
        raise NotImplementedError()

    def forward(self: "Network", x: Any) -> Any:
        raise NotImplementedError()

    def save(
        self: "Network",
        folder_path: str,
        file_name: str,
    ) -> None:
        if not exists(folder_path):
            makedirs(folder_path)

        save(self.state_dict(), join(folder_path, file_name))
