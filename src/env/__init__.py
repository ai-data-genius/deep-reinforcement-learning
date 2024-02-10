class Env:
    def reset(self: "Env"):
        raise NotImplementedError

    def step(self: "Env", action):
        raise NotImplementedError
