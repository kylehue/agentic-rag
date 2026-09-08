from app.plugin.base import Plugin
from app.plugin.context import IngestionContext
from app.plugin.hooks import HookBus


class PluginRegistry:
    """Holds registered plugins and wires their decorated hook handlers into the bus."""

    def __init__(self, hooks: HookBus) -> None:
        self._hooks = hooks
        self._plugins: list[Plugin] = []
        self._plugins_by_name: dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        if plugin.name in self._plugins_by_name:
            raise ValueError(
                f"Plugin name '{plugin.name}' is already registered."
            )

        self._plugins.append(plugin)
        self._plugins_by_name[plugin.name] = plugin

        self._wire_hooks(plugin)

    def _wire_hooks(self, plugin: Plugin) -> None:
        for attr_name, attr in vars(type(plugin)).items():
            if not callable(attr):
                continue
            hook_name = getattr(attr, "__hook_name__", None)
            if isinstance(hook_name, str):
                self._hooks.register(hook_name, getattr(plugin, attr_name))

    def plugin_for(self, name: str) -> Plugin | None:
        return self._plugins_by_name.get(name)

    def accepting_plugins(self, context: IngestionContext) -> list[Plugin]:
        """All registered plugins that want to process this document."""
        return [plugin for plugin in self._plugins if plugin.accepts(context)]
