from collections.abc import Callable


class Registry:
    """Map registered class names to classes."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._classes: dict[str, type] = {}

    def register(self) -> Callable[[type], type]:
        def decorator(cls: type) -> type:
            name = cls.__name__
            if name in self._classes:
                raise KeyError(
                    f"{name!r} is already registered in {self.name}."
                )

            self._classes[name] = cls
            return cls

        return decorator

    def get(self, name: str) -> type:
        if name not in self._classes:
            available = ", ".join(self._classes) or "(none)"
            raise KeyError(
                f"{name!r} is not registered in {self.name}. "
                f"Registered classes: {available}. "
                "Import the module defining the class before using it."
            )

        return self._classes[name]


ACTORS = Registry("actors")
CRITICS = Registry("critics")