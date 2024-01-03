class Env:
    def render(self):
        raise NotImplementedError

    def reset(self: "Env"):
        raise NotImplementedError

    def step(self: "Env", action):
        raise NotImplementedError
