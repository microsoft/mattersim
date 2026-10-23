"""Tests that MatterSimCalculator instances do not share args_dict state."""

from mattersim.forcefield import MatterSimCalculator


class TestCalculatorArgsDict:
    """calculate() writes batch settings into args_dict, so every calculator
    must own its dictionary (see microsoft/mattersim#177)."""

    def test_default_args_dict_is_not_shared(self, mattersim_potential_cpu, si_diamond):
        """Using one calculator must not change another's args_dict."""
        calc1 = MatterSimCalculator(potential=mattersim_potential_cpu, device="cpu")
        calc2 = MatterSimCalculator(potential=mattersim_potential_cpu, device="cpu")
        assert calc1.args_dict is not calc2.args_dict

        si_diamond.calc = calc2
        si_diamond.get_potential_energy()

        assert calc1.args_dict == {}
        assert calc2.args_dict == {"batch_size": 1, "only_inference": 1}

    def test_caller_args_dict_is_not_mutated(self, mattersim_potential_cpu, si_diamond):
        """A dictionary passed by the caller keeps its original contents."""
        args = {"batch_size": 4}
        calc = MatterSimCalculator(
            potential=mattersim_potential_cpu, args_dict=args, device="cpu"
        )

        si_diamond.calc = calc
        si_diamond.get_potential_energy()

        assert args == {"batch_size": 4}
        assert calc.args_dict == {"batch_size": 1, "only_inference": 1}
