import unittest
from t4decode.core import make_cases, make_perf_cases, score_answer, training_sequence, select_configuration

class CoreTests(unittest.TestCase):
    def test_split_reproducibility_and_disjoint_prompts(self):
        groups = [make_cases(split, 30) for split in ("train", "validation", "test")]
        self.assertEqual(groups[0], make_cases("train", 30))
        sets = [{r["prompt"] for r in group} for group in groups]
        for a in range(3):
            for b in range(a):
                self.assertFalse(sets[a] & sets[b])

    def test_gold_scoring_not_teacher_agreement(self):
        for case in make_cases("test", 30):
            import json
            answer = json.dumps(case["gold"]) if isinstance(case["gold"], dict) else case["gold"]
            self.assertEqual(score_answer(case, answer), 1.0)
            self.assertEqual(score_answer(case, "không biết"), 0.0)

    def test_mask_prompt_and_do_not_silently_truncate(self):
        self.assertEqual(training_sequence([1,2], [3,4], 4), dict(input_ids=[1,2,3,4], labels=[-100,-100,3,4]))
        self.assertIsNone(training_sequence([1,2], [3,4], 3))
        self.assertIsNone(training_sequence([1,2], [], 3))

    def test_selector_rejects_oom_incorrect_low_quality_and_partial(self):
        rows = []
        specs = [("fp16", 1.0, 7, 1.0, True), ("int8", .8, 5, 1.0, True),
                 ("nf4", .5, 3, 0.0, True), ("fp16_raw_k4", .4, 13, 1.0, True),
                 ("fp16_raw_dynamic", .3, 8, 1.0, False)]
        for name, time, memory, score, correct in specs:
            for kind in ("quality", "performance"):
                rows.append(dict(candidate=name, id=kind, kind=kind, status="ok", total_s=time,
                    output_tokens=10, peak_vram_gib=memory, peak_ram_gib=2, correctness=correct,
                    score=score if kind == "quality" else None))
        rows.append(dict(candidate="fp16_trained_k4", id="quality", kind="quality", status="ok",
                         total_s=.01, output_tokens=10, peak_vram_gib=2, peak_ram_gib=1, score=1.0))
        policy = select_configuration(rows)
        self.assertEqual(policy["candidate"], "int8")
        self.assertIn("quality_gate", policy["exclusions"]["nf4"])
        self.assertIn("memory_budget", policy["exclusions"]["fp16_raw_k4"])
        self.assertIn("greedy_mismatch", policy["exclusions"]["fp16_raw_dynamic"])
        self.assertIn("incomplete_or_failed", policy["exclusions"]["fp16_trained_k4"])

if __name__ == "__main__":
    unittest.main()
