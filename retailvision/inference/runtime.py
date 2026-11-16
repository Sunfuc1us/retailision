"""Own the single resident CUDA pipeline and its switching lock."""
from threading import Lock


class Runtime:
    def __init__(self):
        self.pipe = None
        self.key = None
        self.lock = Lock()

    def ensure(self, profile, backend):
        config = profile.settings()
        if backend != 'pytorch' and backend not in config.get('deployment', {}):
            raise ValueError(f'{profile.name} has no {backend} export; select PyTorch')
        key = (str(profile.config), backend)
        if self.pipe is None or self.key != key:
            if self.pipe is not None:
                self.pipe.unload()
            self.pipe = None
            self.key = None
            from retailvision.inference.factory import load_pipeline
            try:
                pipe = load_pipeline(str(profile.config), backend)
            except Exception as error:
                raise RuntimeError(str(error)) from error
            pipe.model_id = profile.id
            pipe.model_name = profile.name
            pipe.compact_masks = True
            self.pipe, self.key = pipe, key
        return self.pipe

    def info(self):
        if self.pipe is None:
            return dict(loaded=False)
        return dict(loaded=True, **self.pipe.get_runtime_info())
