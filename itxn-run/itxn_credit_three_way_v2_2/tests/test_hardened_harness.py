import copy
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features import get_customer_profile, all_profiles
from agent_core import simulated_agent_payload
from harness import validate_payload, HarnessValidationError


class HardenedHarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = get_customer_profile("SYN-CUST-0001")
        cls.good = simulated_agent_payload(cls.profile)
        profiles, _ = all_profiles()
        cls.reversal_profile = next(p for p in profiles if "REVERSALS_OBSERVED" in p["available_signals"])
        cls.reversal_payload = simulated_agent_payload(cls.reversal_profile)

    def test_valid_complete_payload_passes(self):
        self.assertEqual(validate_payload(self.good, self.profile)["status"], "PASS")

    def test_unsupported_credit_claim_rejected(self):
        bad = copy.deepcopy(self.good)
        bad["narrative"] = "The customer is low risk; approve the loan."
        with self.assertRaises(HarnessValidationError):
            validate_payload(bad, self.profile)

    def test_persian_credit_claim_rejected(self):
        bad = copy.deepcopy(self.good)
        bad["narrative"] = "این مشتری کم‌ریسک است و وام بدهید."
        with self.assertRaises(HarnessValidationError):
            validate_payload(bad, self.profile)

    def test_omitted_observed_reversal_signal_rejected(self):
        bad = copy.deepcopy(self.reversal_payload)
        bad["selected_signals"].remove("REVERSALS_OBSERVED")
        with self.assertRaises(HarnessValidationError):
            validate_payload(bad, self.reversal_profile)

    def test_extra_recommendation_key_rejected(self):
        bad = copy.deepcopy(self.good)
        bad["recommendation"] = "APPROVE"
        with self.assertRaises(HarnessValidationError):
            validate_payload(bad, self.profile)

    def test_malformed_signal_type_rejected_cleanly(self):
        bad = copy.deepcopy(self.good)
        bad["selected_signals"] = [{}]
        with self.assertRaises(HarnessValidationError):
            validate_payload(bad, self.profile)

    def test_wrong_customer_rejected(self):
        bad = copy.deepcopy(self.good)
        bad["customer_id"] = "SYN-CUST-9999"
        with self.assertRaises(HarnessValidationError):
            validate_payload(bad, self.profile)

    def test_unexpected_missing_evidence_code_rejected(self):
        bad = copy.deepcopy(self.good)
        bad["missing_evidence"] = bad["missing_evidence"] + ["FAKE_DATA_SOURCE"]
        with self.assertRaises(HarnessValidationError):
            validate_payload(bad, self.profile)


if __name__ == "__main__":
    unittest.main()
