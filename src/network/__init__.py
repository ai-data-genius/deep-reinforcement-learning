from os import makedirs
from os.path import exists, join

from torch import save
from torch.nn import Module


class Network(Module):
    def forward(self: "Network"):
        raise NotImplementedError

    def save(
        self: "Network",
        folder_path: str,
        file_name: str,
    ) -> None:
        if not exists(folder_path):
            makedirs(folder_path)

        save(self.state_dict(), join(folder_path, file_name))
