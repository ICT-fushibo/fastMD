"""Prophet names reach the optional loader without importing its framework."""
import unittest
from unittest.mock import patch

from fastmd.adapters import load_adapter
from fastmd.adapters.negotiate import NegotiationError


class FoundationModelLoadingTests(unittest.TestCase):
    def test_public_names_dispatch_with_all_loading_options(self):
        for family, names in (
            ("prophet", ("Prophet-OAME-MBD", "prophet_oame_mbd")),
        ):
            for name in names:
                with self.subTest(name=name), patch(f"fastmd.adapters.{family}.load_{family}") as loader:
                    result = load_adapter(name, model_path="/checkpoint", device="cpu",
                                          model_dtype="float32", optimization_profile="generic")
                    self.assertIs(result, loader.return_value)
                    loader.assert_called_once_with(name, model_path="/checkpoint", device="cpu",
                                                   model_dtype="float32", optimization_profile="generic")

    def test_legacy_profile_rejected_before_loading(self):
        for family, name in (("prophet", "Prophet-OAME-MBD"),):
            with self.subTest(name=name), patch(f"fastmd.adapters.{family}.load_{family}") as loader:
                with self.assertRaises(NegotiationError):
                    load_adapter(name, model_path="/checkpoint", optimization_profile="legacy")
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
