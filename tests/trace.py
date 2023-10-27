import unittest

import torch

import baukit
import tests.utils.functions as utils
from tests.utils.models import DummyModel


class TestTrace(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.hidden_dims = (64, 64, 64, 64)
        self.model = DummyModel()
        self.model.eval()

    @torch.inference_mode()
    def test_0_hooking_module(self):
        module = baukit.get_module(self.model, "layers.block_1")
        assert module is not None, "Error hooking module"
        assert module.linear.weight.shape == (
            self.hidden_dims[1],
            self.hidden_dims[2],
        ), "Shape of module's weight is not correct"

        print(". PASSED: module hooked successfully")

    @torch.inference_mode()
    def test_1_reading_output(self):
        module_0 = baukit.get_module(self.model, "layers.block_0")
        embedding = torch.randn(15, self.model.input_dim).to(
            device=self.model.device, dtype=self.model.dtype
        )
        expected_output = module_0(embedding)
        # hook the output of `layers.block_0` in a forward pass
        with baukit.Trace(
            self.model, layer="layers.block_0", retain_output=True
        ) as trace:
            self.model(embedding)
        assert torch.allclose(
            expected_output, trace.output
        ), f"Error reading output, output is not correct, L2 error: {torch.norm(expected_output - trace.output)}"
        print(" PASSED: output read successfully")

    @torch.inference_mode()
    def test_2_reading_input(self):
        self.model.pass_as_kwargs = False
        module_0 = baukit.get_module(self.model, "layers.block_0")
        embedding = torch.randn(15, self.model.input_dim).to(
            device=self.model.device, dtype=self.model.dtype
        )
        expected_input = module_0(embedding)
        # output of `layers.block_0` is passed as input to `layers.block_1`
        with baukit.Trace(
            self.model, layer="layers.block_1", retain_input=True
        ) as trace:
            self.model(embedding)
        assert (
            type(trace.input) == torch.Tensor
        ), f"expected torch.Tensor, `trace.input` is of type: {type(trace.input)}"
        assert torch.allclose(
            expected_input, trace.input
        ), f"input is not correct, L2 error: {torch.norm(expected_input - trace.input)}"
        print(" PASSED: input read successfully")

    @torch.inference_mode()
    def test_3_reading_input(self):
        self.model.pass_as_kwargs = True
        module_0 = baukit.get_module(self.model, "layers.block_0")
        embedding = torch.randn(15, self.model.input_dim).to(
            device=self.model.device, dtype=self.model.dtype
        )
        expected_input = module_0(embedding)
        # output of `layers.block_0` is passed as input to `layers.block_1`
        with baukit.Trace(
            self.model, layer="layers.block_1", retain_input=True
        ) as trace:
            self.model(embedding)
        assert (
            type(trace.input_kw) == dict
        ), f"expecting a dictionary. `trace.input_kw` is of type: {type(trace.input)}"
        assert set(trace.input_kw.keys()) == {
            "hidden_states",
            "scale_factor",
        }, f"`trace.input_kw` is expected to have only two keys: `hidden_states` and `scale_factor`, found {trace.input_kw.keys()}"
        assert torch.allclose(
            expected_input, trace.input_kw["hidden_states"]
        ), f"input_kw is not correct, L2 error: {torch.norm(expected_input - trace.input)}"
        print(" PASSED: input_kw read successfully")

    @torch.inference_mode()
    def test_4_intervention(self):
        # first run with random to store patching
        embedding = torch.randn(15, self.model.input_dim).to(
            device=self.model.device, dtype=self.model.dtype
        )
        intervention_layer = "layers.block_1"
        intervene_at = 5
        with baukit.Trace(
            self.model,
            layer=intervention_layer,
        ) as trace:
            expected_output = self.model(embedding)
        patching = trace.output[intervene_at]

        # now run with intervention on a different input
        new_embedding = torch.randn(15, self.model.input_dim).to(
            device=self.model.device, dtype=self.model.dtype
        )
        intervention_layer = "layers.block_1"
        with baukit.Trace(
            self.model,
            layer=intervention_layer,
            edit_output=utils.intervention(intervention_layer, intervene_at, patching),
        ) as trace:
            new_output = self.model(new_embedding)

        assert not torch.allclose(
            expected_output, new_output
        ), "both outputs are the same"
        assert torch.allclose(
            expected_output[intervene_at], new_output[intervene_at]
        ), "Error in intervention, output is not correct"

        print(" PASSED: intervention successful")

    def test_5_early_exit(self):
        embedding = torch.randn(15, self.model.input_dim).to(
            device=self.model.device, dtype=self.model.dtype
        )
        with baukit.Trace(
            self.model, layer="layers.block_1", retain_input=True, stop=True
        ) as trace:
            output = self.model(embedding)

        # ! Is this a good test for this?
        # the forward will stop as soon as the output of `layers.block_1` is collected
        # `output` variable should never exist
        assert "output" not in locals(), "Error in early exit"
        print(" PASSED: early exit successful")


if __name__ == "__main__":
    unittest.main()
