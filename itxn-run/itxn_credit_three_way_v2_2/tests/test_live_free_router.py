"""No-network tests for low-cost live-provider configuration and request budget."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import agent_core
from features import get_customer_profile


class FreeRouterTests(unittest.TestCase):
    def test_openrouter_defaults_to_free_router(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": "https://openrouter.ai/api/v1"}, clear=True):
            self.assertEqual(agent_core._requested_model(), "openrouter/free")

    def test_non_openrouter_requires_explicit_model(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": "https://example.invalid/v1"}, clear=True):
            with self.assertRaises(RuntimeError):
                agent_core._requested_model()

    def test_key_for_openrouter_does_not_accidentally_use_openai_key(self):
        with patch.dict(os.environ, {"LLM_BASE_URL": "https://openrouter.ai/api/v1", "OPENAI_API_KEY": "wrong-provider-key"}, clear=True):
            self.assertEqual(agent_core._api_key(), "")

    def test_live_agent_uses_one_tool_then_one_summary_request(self):
        profile = get_customer_profile("SYN-CUST-0001")
        tool_response = {
            "id": "req-tool", "model": "provider/demo-free-model",
            "choices": [{"message": {
                "content": None,
                "tool_calls": [{"id": "call-1", "type": "function", "function": {
                    "name": "get_customer_profile", "arguments": json.dumps({"customer_id": "SYN-CUST-0001"})
                }}]
            }}], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        }
        summary = {
            "id": "req-summary", "model": "provider/demo-free-model",
            "choices": [{"message": {"content": json.dumps(agent_core.simulated_agent_payload(profile))}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 70, "total_tokens": 270}
        }
        with patch.object(agent_core, "_post_chat", side_effect=[tool_response, summary]) as post:
            with patch.dict(os.environ, {"LLM_BASE_URL": "https://openrouter.ai/api/v1"}, clear=True):
                result = agent_core.run_live_agent("SYN-CUST-0001", profile_override=profile)
        self.assertEqual(post.call_count, 2)
        sent_tool_choice = post.call_args_list[0].args[0]["tool_choice"]
        self.assertEqual(sent_tool_choice, {"type": "function", "function": {"name": "get_customer_profile"}})
        self.assertEqual(result["model_requests"], 2)
        self.assertEqual(result["models_used"], ["provider/demo-free-model"])
        self.assertEqual(result["usage"]["total_tokens"], 390)
        self.assertEqual(result["payload"]["customer_id"], "SYN-CUST-0001")

    def test_live_comparison_reuses_one_payload_without_network(self):
        import tempfile
        from unittest.mock import patch
        import run_live_comparison
        profile = get_customer_profile("SYN-CUST-0001")
        payload = agent_core.simulated_agent_payload(profile)
        fake_run = {
            "payload": payload, "profile": profile, "runtime": 0.01,
            "model_requests": 2, "model": "openrouter/free",
            "model_requested": "openrouter/free", "models_used": ["provider/demo-free-model"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "request_ids": ["req1", "req2"], "engine": "mock-live-agent",
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "live_comparison.json"
            with patch.object(run_live_comparison, "run_live_agent", return_value=fake_run) as live:
                result = run_live_comparison.run("SYN-CUST-0001", output_path=output)
            self.assertEqual(live.call_count, 1)
            self.assertTrue(output.exists())
            self.assertEqual(result["agent_plus_harness"]["harness_status"], "ACCEPTED")
            self.assertTrue(result["agent_plus_harness"]["same_agent_payload_reused"])
            self.assertEqual(result["agent"]["api_requests"], 2)


if __name__ == "__main__":
    unittest.main()
