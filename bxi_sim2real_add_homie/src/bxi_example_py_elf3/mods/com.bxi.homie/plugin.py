"""HOMIE Mod entrypoint."""

from bxi_example_py_elf3.framework.mod_api import (
    ModDefinition,
    ModLoadContext,
    ResourceKey,
    ResourceLoadContext,
)
from bxi_example_py_elf3.policies import HomiePolicy

from .state import HomieState


HOMIE_POLICY = ResourceKey[HomiePolicy]("com.bxi.homie/policy")


def _load_policy(context: ResourceLoadContext) -> HomiePolicy:
    return HomiePolicy(
        str(context.asset("assets/homie_elf3.onnx")),
        backend="onnxruntime",
    )


def create_mod(context: ModLoadContext) -> ModDefinition:
    context.register_resource(HOMIE_POLICY, _load_policy, policy="on_demand")
    policy = context.resource(HOMIE_POLICY)
    return ModDefinition(
        state_factories={
            "homie": lambda state: HomieState(
                state.name,
                state.state_id,
                policy,
            )
        }
    )


__all__ = ["HOMIE_POLICY", "create_mod"]
