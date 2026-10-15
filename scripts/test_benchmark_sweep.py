"""Regression checks for the notebook's budget mapping and result handling."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from benchmark_inference import sweep_cases, run_sweep
from benchmark_notebook import run_model, summary_table


class SweepTests(unittest.TestCase):
    def test_equal_nfe_budget_mapping(self):
        budgets = [1, 2, 4, 8, 10, 16, 32]
        self.assertEqual(sweep_cases('act', budgets), [('ACT', None)])
        self.assertEqual(sweep_cases('dp3', budgets), [('DDIM', n) for n in budgets])
        for model in ('fm', 'fm3'):
            cases = sweep_cases(model, budgets)
            self.assertEqual(len(cases), 13)
            self.assertEqual([n * 2 for m, n in cases if m == 'heun'], budgets[1:])
        with self.assertRaises(ValueError):
            sweep_cases('dp', [0])

    def test_sweep_loads_once_and_reseeds_each_budget(self):
        policy = SimpleNamespace(num_inference_steps=1, integration_method='euler')
        seen = []
        def measure(kind, checkpoint, prepared, args, device, seed):
            seen.append((policy.integration_method, policy.num_inference_steps, seed))
            return {'mean_ms': 1., 'timings_ms': [1.], 'model': kind}
        with tempfile.TemporaryDirectory() as tmp, \
             patch('benchmark_inference.prepare', return_value=(policy,)) as prepare, \
             patch('benchmark_inference.measure', side_effect=measure):
            spec = dict(model='fm', checkpoint='unused', episode='unused', device='cpu',
                        states=1, warmup=1, iterations=1, repeats=2, threads=1,
                        nfe_budgets=[1, 2], seed=42, output=str(Path(tmp) / 'result.json'))
            run_sweep(spec)
            prepare.assert_called_once()
            self.assertEqual(len(seen), 6)
            self.assertEqual([s for _, _, s in seen], [42, 42, 42, 43, 43, 43])
            self.assertNotEqual(seen[0][:2], seen[3][:2])

    def test_summary_pools_samples_instead_of_averaging_percentiles(self):
        rows = [dict(model='dp', sampler='DDIM', steps=2, unet_evaluations=2,
                     timings_ms=times, mean_ms=np.mean(times), peak_cpu_rss_mb=100.,
                     peak_cuda_allocated_mb=None) for times in ([1., 2.], [3., 100.])]
        result = summary_table({'dp': rows}).iloc[0]
        self.assertAlmostEqual(result.p95_ms, np.percentile([1., 2., 3., 100.], 95))
        self.assertEqual(result.repeats, 2)

    def test_failed_worker_preserves_partial_results_and_parent(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp); (repo / 'scripts').mkdir()
            worker = repo / 'scripts/benchmark_inference.py'
            worker.write_text('import json,sys\nfrom pathlib import Path\n'
                              'spec=json.loads(Path(sys.argv[2]).read_text())\n'
                              'Path(spec["output"]).write_text(json.dumps({"results":[{"saved":True}]}))\n'
                              'print("simulated failure")\nsys.exit(9)\n')
            (repo / 'model.ckpt').touch(); (repo / 'episode_0.hdf5').touch()
            rows = run_model(repo, 'dp', repo / 'model.ckpt', repo,
                             output_dir=repo / 'outputs')
            self.assertEqual(rows, [{'saved': True}])
            report = json.loads(next((repo / 'outputs').glob('*/results.json')).read_text())
            self.assertEqual(report['worker_exit_code'], 9)
            self.assertEqual(report['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
