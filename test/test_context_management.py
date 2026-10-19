import unittest

from core.context_management import (
    Handoff,
    RoleContextManager,
    application_cache_key,
    context_scope,
    current_context,
    role_context,
    stable_prefix_fingerprint,
)


class RoleContextManagerTests(unittest.TestCase):
    def test_role_histories_are_private_but_task_facts_are_shared(self):
        manager = RoleContextManager()
        manager.add_turn("alice", "task-1", "teacher", prompt="p1", response="r1")
        manager.add_turn("alice", "task-1", "validator", prompt="p2", response="r2")
        manager.put_fact("alice", "task-1", "goal", "solve")
        self.assertEqual(manager.role_history("alice", "task-1", "teacher"), [{"prompt": "p1", "response": "r1"}])
        self.assertEqual(manager.role_history("alice", "task-1", "validator"), [{"prompt": "p2", "response": "r2"}])
        self.assertEqual(manager.task_facts("alice", "task-1")["goal"], "solve")

    def test_user_and_task_isolation(self):
        manager = RoleContextManager()
        manager.put_fact("alice", "task-1", "secret", "A")
        self.assertEqual(manager.task_facts("bob", "task-1"), {})
        self.assertEqual(manager.task_facts("alice", "task-2"), {})

    def test_handoffs_are_explicit_and_filterable(self):
        manager = RoleContextManager()
        record = Handoff("task-1", "reasoner", "teacher", "answer-v2", {"answer": "42"}, ("evidence-1",))
        manager.handoff("alice", record)
        self.assertEqual(manager.handoffs("alice", "task-1", target_role="teacher"), [record])
        self.assertEqual(manager.handoffs("alice", "task-1", target_role="validator"), [])

    def test_scope_restores_parent_after_role_call(self):
        @role_context("teacher")
        def current_role():
            return current_context()
        with context_scope("alice", "task-1", "orchestrator"):
            nested = current_role()
            self.assertEqual((nested.user_id, nested.task_id, nested.role_id), ("alice", "task-1", "teacher"))
            self.assertEqual(current_context().role_id, "orchestrator")
        self.assertIsNone(current_context())

    def test_cache_keys_change_for_security_or_semantic_inputs(self):
        base = dict(user_id="alice", task_id="task", role_id="teacher", provider_id="p", model_id="m",
                    prompt_version="v1", input_digest="abc")
        key = application_cache_key(**base)
        self.assertEqual(key, application_cache_key(**base))
        self.assertNotEqual(key, application_cache_key(**{**base, "user_id": "bob"}))
        self.assertNotEqual(key, application_cache_key(**{**base, "prompt_version": "v2"}))
        self.assertNotEqual(key, application_cache_key(**{**base, "evidence_version": "new"}))

    def test_stable_prefix_fingerprint_ignores_no_dynamic_data_by_construction(self):
        self.assertEqual(stable_prefix_fingerprint("system", "role", version="v1"),
                         stable_prefix_fingerprint("system", "role", version="v1"))
        self.assertNotEqual(stable_prefix_fingerprint("system", "role", version="v1"),
                            stable_prefix_fingerprint("system", "role", version="v2"))

    def test_history_is_bounded(self):
        manager = RoleContextManager(history_limit=2)
        for index in range(4):
            manager.add_turn("alice", "task", "teacher", prompt=str(index), response=str(index))
        self.assertEqual([turn["prompt"] for turn in manager.role_history("alice", "task", "teacher")], ["2", "3"])


if __name__ == "__main__":
    unittest.main()
