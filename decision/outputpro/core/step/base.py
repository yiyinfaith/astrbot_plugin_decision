from abc import ABC, abstractmethod

from ..config import PluginConfig
from ..model import OutContext, StepName, StepResult


class BaseStep(ABC):
    """
    所有步骤的基类。
    子类必须实现 handle()
    """

    #: 步骤名（必须覆盖）
    name: StepName

    def __init__(self, config: PluginConfig):
        self.plugin_config = config

    @abstractmethod
    async def handle(self, ctx: OutContext) -> StepResult:
        """
        处理单次步骤的核心逻辑。

        参数
        ----
        ctx : OutContext
            上游传递的上下文对象，只读。

        返回
        ----
        None
            如需向下游传递数据，请通过 ctx 的字段或显式返回值，
            并在类文档中说明约定。
        """
        ...  # 子类必须覆盖此处

    async def initialize(self) -> None: ...
    async def terminate(self) -> None: ...
