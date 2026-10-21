import sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from features import load_rows, calculate_features, make_profile, get_customer_profile
from agent_core import simulated_agent_payload
from harness import validate_payload, HarnessValidationError, audit_hash


class FeatureTests(unittest.TestCase):
    def test_known_two_row_calculation(self):
        rows = [
            {"synthetic_customer_id": "C1", "txnamt": "100", "datein": "2026-01-01", "respcode": "00", "rev_flag": "0"},
            {"synthetic_customer_id": "C1", "txnamt": "50", "datein": "2026-01-01", "respcode": "05", "rev_flag": "1"},
        ]
        f = calculate_features(rows)[0]
        self.assertEqual(f["txn_count"], 2)
        self.assertEqual(f["active_days_observed"], 1)
        self.assertEqual(f["amount_sum_observed"], 150.0)
        self.assertEqual(f["reversal_count"], 1)
        self.assertEqual(f["success_rate_among_known_codes"], 0.5)

    def test_missing_required_column_fails(self):
        with self.assertRaises(ValueError):
            load_rows(ROOT / "tests" / "fixtures" / "missing_column.csv")

    def test_blank_customer_id_fails(self):
        rows = [{"synthetic_customer_id": " ", "txnamt": "10"}]
        with self.assertRaises(ValueError):
            calculate_features(rows)


class HarnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = get_customer_profile("SYN-CUST-0001")
        cls.good = simulated_agent_payload(cls.profile)

    def test_valid_demo_payload_passes(self):
        self.assertEqual(validate_payload(self.good, self.profile)["status"], "PASS")

    def test_unknown_signal_rejected(self):
        bad = dict(self.good); bad["selected_signals"] = self.good["selected_signals"] + ["INVENTED_RISK_SIGNAL"]
        with self.assertRaises(HarnessValidationError): validate_payload(bad, self.profile)

    def test_unsafe_credit_outcome_rejected(self):
        bad = dict(self.good); bad["review_outcome"] = "APPROVED"
        with self.assertRaises(HarnessValidationError): validate_payload(bad, self.profile)

    def test_unknown_evidence_field_rejected(self):
        bad = dict(self.good); bad["evidence_fields"] = self.good["evidence_fields"] + ["imaginary_score"]
        with self.assertRaises(HarnessValidationError): validate_payload(bad, self.profile)

    def test_missing_repayment_gaps_rejected(self):
        bad = dict(self.good); bad["missing_evidence"] = ["REPAYMENT_HISTORY"]
        with self.assertRaises(HarnessValidationError): validate_payload(bad, self.profile)

    def test_numerical_claim_in_narrative_rejected(self):
        bad = dict(self.good); bad["narrative"] = "This customer made 35 transactions."
        with self.assertRaises(HarnessValidationError): validate_payload(bad, self.profile)

    def test_audit_hash_is_stable(self):
        self.assertEqual(audit_hash({"b": 2, "a": 1}), audit_hash({"a": 1, "b": 2}))


if __name__ == "__main__":
    unittest.main()
