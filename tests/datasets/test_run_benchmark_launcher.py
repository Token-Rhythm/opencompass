import json
import os
import shlex
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'script/run_benchmarks.sh'
TWO_HOUR_SCRIPT = ROOT / 'script/run_posttrain_2h.sh'
CORE_SCRIPT = ROOT / 'script/run_posttrain_objective_benchmark.sh'


def run_script(*args, env=None):
    command = [
        'bash', str(SCRIPT), '--model', 'test-model', '--base-url',
        'http://127.0.0.1:8000/v1', '--tokenizer-path',
        'test/tokenizer', *args
    ]
    return subprocess.run(command,
                          cwd=ROOT,
                          text=True,
                          capture_output=True,
                          env=env)


def test_default_dry_run_selects_every_benchmark_once(tmp_path):
    result = run_script('--dry-run', '--run-id', 'test_all',
                        '--output-root', str(tmp_path))

    assert result.returncode == 0, result.stderr
    assert 'Selected benchmarks (20):' in result.stdout
    assert result.stdout.count('DRY-RUN:') == 20
    assert result.stdout.count('DRY-RUN score gate:') == 20
    assert 'Execution policy: sequential infer -> eval -> numeric-summary check' in result.stdout
    assert 'aa_lcr: infer -> eval -> summary' not in result.stdout
    assert 'multichallenge: infer -> eval -> summary' not in result.stdout


def test_partial_selection_accepts_aliases_and_deduplicates(tmp_path):
    result = run_script('--dry-run', '--run-id', 'test_partial',
                        '--output-root', str(tmp_path), '--benchmark',
                        'GPQA,longbenchv2', '--benchmark',
                        'gpqa_diamond,HMMT-Feb-2025')

    assert result.returncode == 0, result.stderr
    assert ('Selected benchmarks (3): gpqa_diamond hmmt_feb_2025 '
            'longbench_v2') in result.stdout
    assert result.stdout.count('DRY-RUN:') == 3
    assert ('run_posttrain_objective_benchmark.sh hmmt_feb_2025'
            in result.stdout)


def test_priority_suite_profiles_regular_then_hmmt_then_longbench(tmp_path):
    selected = (
        "ceval,ifeval,mmmlu_downsampling,gpqa_diamond,ifbench,aime_2025,"
        "aime_2026,livecodebench,mmlu_redux_downsampling,"
        "mmlu_prox_downsampling,global_piqa_downsampling,"
        "include_downsampling,hmmt_feb_2026,longbench_v2"
    )
    result = run_script(
        '--dry-run', '--run-id', 'test_priority_suite',
        '--output-root', str(tmp_path), '--benchmark', selected,
        '--max-seq-len', '65536', '--max-out-len', '32768',
        '--batch-size', '512', '--max-workers', '512',
        '--query-per-second', '128')

    assert result.returncode == 0, result.stderr
    expected = (
        'Selected benchmarks (14): ceval ifeval mmmlu_downsampling '
        'gpqa_diamond ifbench aime_2025 aime_2026 livecodebench '
        'mmlu_redux_downsampling mmlu_prox_downsampling '
        'global_piqa_downsampling include_downsampling hmmt_feb_2026 '
        'longbench_v2'
    )
    assert expected in result.stdout
    assert result.stdout.index('include_downsampling: infer -> eval -> summary') < result.stdout.index(
        'hmmt_feb_2026: infer -> eval -> summary') < result.stdout.index(
            'longbench_v2: infer -> eval -> summary')

    commands = {
        line.split('run_posttrain_objective_benchmark.sh ', 1)[1].split()[0]:
        shlex.split(line.removeprefix('DRY-RUN:'))
        for line in result.stdout.splitlines()
        if line.startswith('DRY-RUN:')
    }

    def final_value(command, option):
        positions = [index for index, value in enumerate(command) if value == option]
        assert positions, option
        return command[positions[-1] + 1]

    regular = commands['ceval_evalscope']
    assert final_value(regular, '--max-seq-len') == '65536'
    assert final_value(regular, '--max-out-len') == '32768'
    assert final_value(regular, '--batch-size') == '19'
    assert final_value(regular, '--max-workers') == '19'
    assert final_value(regular, '--dataset-workers') == '14'

    hmmt = commands['hmmt2026']
    assert final_value(hmmt, '--max-seq-len') == '131072'
    assert final_value(hmmt, '--max-out-len') == '81920'
    assert final_value(hmmt, '--batch-size') == '128'
    assert final_value(hmmt, '--max-workers') == '128'
    assert final_value(hmmt, '--query-per-second') == '128'
    assert final_value(hmmt, '--dataset-workers') == '1'

    longbench = commands['longbenchv2']
    assert final_value(longbench, '--max-seq-len') == '262144'
    assert final_value(longbench, '--max-out-len') == '32768'
    assert final_value(longbench, '--batch-size') == '32'
    assert final_value(longbench, '--max-workers') == '32'
    assert final_value(longbench, '--dataset-workers') == '1'


def test_objective_tasks_call_the_core_launcher_directly(tmp_path):
    result = run_script('--dry-run', '--run-id', 'test_direct_core',
                        '--output-root', str(tmp_path), '--benchmark',
                        'mmlu_pro,ceval,longbench_v2')

    assert result.returncode == 0, result.stderr
    command_lines = [line for line in result.stdout.splitlines()
                     if line.startswith('DRY-RUN:')]
    assert len(command_lines) == 3
    assert all('run_posttrain_objective_benchmark.sh' in line
               for line in command_lines)
    assert all('run_mmlu_pro.sh' not in line for line in command_lines)
    assert all('run_ceval_evalscope.sh' not in line for line in command_lines)
    assert all('run_longbenchv2.sh' not in line for line in command_lines)
    assert any('run_posttrain_objective_benchmark.sh mmlu_pro' in line
               for line in command_lines)
    assert any('run_posttrain_objective_benchmark.sh ceval_evalscope' in line
               for line in command_lines)


def test_external_judge_benchmarks_are_not_launcher_targets(tmp_path):
    for benchmark in ('aa_lcr', 'multichallenge'):
        result = run_script(
            '--dry-run', '--output-root', str(tmp_path),
            '--benchmark', benchmark)

        assert result.returncode == 2
        assert f'unknown benchmark: {benchmark}' in result.stderr
        assert 'DRY-RUN:' not in result.stdout


def test_unknown_benchmark_fails_before_any_launch(tmp_path):
    result = run_script('--dry-run', '--output-root', str(tmp_path),
                        '--benchmark', 'not-a-benchmark')

    assert result.returncode == 2
    assert 'unknown benchmark: not-a-benchmark' in result.stderr
    assert 'DRY-RUN:' not in result.stdout


def test_api_key_is_redacted_from_dry_run(tmp_path):
    result = run_script('--dry-run', '--run-id', 'test_secret',
                        '--output-root', str(tmp_path), '--benchmark', 'ifeval',
                        '--api-key', 'do-not-print-this')

    assert result.returncode == 0, result.stderr
    assert 'do-not-print-this' not in result.stdout
    assert '<redacted>' in result.stdout


def test_agent_eval_upload_is_planned_immediately_after_score_gate(tmp_path):
    result = run_script(
        '--dry-run', '--run-id', 'test_upload', '--output-root', str(tmp_path),
        '--benchmark', 'mmlu_pro,ifeval', '--agent-eval-upload',
        '--agent-eval-evidence-url',
        'https://artifacts.example/evals/{benchmark}/{run_id}')

    assert result.returncode == 0, result.stderr
    assert result.stdout.count('DRY-RUN post-score upload:') == 2
    assert result.stdout.count('opencompass_agent_eval_adapter.py') == 2
    assert result.stdout.count('--mode ingest') == 2
    assert 'direct canonical ingest' in result.stdout
    lines = result.stdout.splitlines()
    score_index = next(index for index, line in enumerate(lines)
                       if line.startswith('DRY-RUN score gate:'))
    upload_index = next(index for index, line in enumerate(lines)
                        if 'opencompass_agent_eval_adapter.py' in line)
    assert score_index < upload_index


def test_agent_eval_review_mode_remains_explicitly_available(tmp_path):
    result = run_script(
        '--dry-run', '--run-id', 'test_review_upload',
        '--output-root', str(tmp_path), '--benchmark', 'mmlu_pro',
        '--agent-eval-upload', '--agent-eval-mode', 'review',
        '--agent-eval-evidence-url',
        'https://artifacts.example/evals/{benchmark}/{run_id}')

    assert result.returncode == 0, result.stderr
    assert '--mode review' in result.stdout
    assert 'pending-review upload' in result.stdout


def test_agent_eval_upload_requires_evidence_url(tmp_path):
    env = os.environ.copy()
    env.pop('AGENT_EVAL_EVIDENCE_URL', None)
    result = run_script('--dry-run', '--output-root', str(tmp_path),
                        '--benchmark', 'mmlu_pro', '--agent-eval-upload',
                        env=env)

    assert result.returncode == 2
    assert '--agent-eval-evidence-url' in result.stderr


def test_hmmt_2025_individual_selectors_are_available():
    launcher = (ROOT /
                'script/run_posttrain_objective_benchmark.sh').read_text()

    assert 'DATASETS_EXPR="[hmmt_2025_datasets[0]]"' in launcher
    assert 'DATASETS_EXPR="[hmmt_2025_datasets[1]]"' in launcher


def test_shared_core_keeps_include_on_fixed_logprob_protocol():
    launcher = (ROOT /
                'script/run_posttrain_objective_benchmark.sh').read_text()

    assert 'mmlu_redux|mmlu_redux_downsampling)' in launcher
    assert 'hmmt_feb_2025)' in launcher
    assert 'hmmt_nov_2025)' in launcher
    assert 'include|include_downsampling)' in launcher
    assert 'PROTOCOL="logprob"' in launcher
    assert "generation_endpoint='completions'" in launcher
    assert 'completion_extra_body={}' in launcher
    assert 'temperature=0' in launcher
    assert "dataset_abbrs=[include_group_name, *include_subsets]" in launcher


def test_former_language_tasks_use_the_shared_core_and_validated_sampling(
        tmp_path):
    result = run_script('--dry-run', '--run-id', 'test_language_defaults',
                        '--output-root', str(tmp_path), '--benchmark',
                        "mmlu_redux_downsampling,hmmt_feb_2025,hmmt_nov_2025,"
                        "include_downsampling",
                        '--temperature', '0.8', '--extra-body-json',
                        '{"top_p":0.9}')

    assert result.returncode == 0, result.stderr
    assert result.stdout.count('--model test-model') == 4
    assert 'run_language_benchmarks_smoke.sh' not in result.stdout
    redux_command = next(line for line in result.stdout.splitlines()
                         if ('run_posttrain_objective_benchmark.sh mmlu_redux_downsampling'
                             in line))
    assert '--max-out-len' not in redux_command
    assert '--temperature 0.8' in redux_command
    assert '--extra-body-json' in redux_command
    assert 'top_p' in redux_command and '0.9' in redux_command
    include_command = next(line for line in result.stdout.splitlines()
                           if ('run_posttrain_objective_benchmark.sh include_downsampling'
                               in line))
    assert '--temperature 0.8' in include_command


def test_all_benchmarks_use_one_shared_execution_script(tmp_path):
    result = run_script('--dry-run', '--run-id', 'test_one_core',
                        '--output-root', str(tmp_path))

    assert result.returncode == 0, result.stderr
    command_lines = [line for line in result.stdout.splitlines()
                     if line.startswith('DRY-RUN:')]
    assert len(command_lines) == 20
    assert all('run_posttrain_objective_benchmark.sh' in line
               for line in command_lines)
    assert all('run_language_benchmarks_smoke.sh' not in line
               for line in command_lines)




def test_model_and_base_url_are_required_before_launch():
    result = subprocess.run(['bash', str(SCRIPT), '--dry-run'],
                            cwd=ROOT,
                            text=True,
                            capture_output=True)

    assert result.returncode == 2
    assert '--model is required' in result.stderr


def test_global_sample_and_stream_idle_options_reach_shared_core(tmp_path):
    result = run_script('--dry-run', '--run-id', 'test_passthrough',
                        '--output-root', str(tmp_path), '--benchmark',
                        'longbench_v2', '--samples', '17',
                        '--stream-idle-timeout', '901')

    assert result.returncode == 0, result.stderr
    command = next(line for line in result.stdout.splitlines()
                   if line.startswith('DRY-RUN:'))
    assert '--samples 17' in command
    assert '--stream-idle-timeout 901' in command


def test_dataset_worker_option_reaches_shared_core(tmp_path):
    result = run_script('--dry-run', '--run-id', 'test_dataset_workers',
                        '--output-root', str(tmp_path), '--benchmark', 'ceval',
                        '--dataset-workers', '32')

    assert result.returncode == 0, result.stderr
    command = next(line for line in result.stdout.splitlines()
                   if line.startswith('DRY-RUN:'))
    assert '--dataset-workers 32' in command


def test_partition_workers_match_dataset_shapes(tmp_path):
    result = run_script(
        '--dry-run', '--run-id', 'test_partition_workers',
        '--output-root', str(tmp_path), '--benchmark',
        'ceval,mmmlu_downsampling,mmlu_prox_downsampling,'
        'mmlu_redux_downsampling,global_piqa_downsampling,'
        'include_downsampling')

    assert result.returncode == 0, result.stderr
    commands = {
        line.split('run_posttrain_objective_benchmark.sh ', 1)[1].split()[0]:
        shlex.split(line.removeprefix('DRY-RUN:'))
        for line in result.stdout.splitlines()
        if line.startswith('DRY-RUN:')
    }

    def final_value(command, option):
        positions = [index for index, value in enumerate(command)
                     if value == option]
        assert positions, option
        return command[positions[-1] + 1]

    assert final_value(commands['ceval_evalscope'],
                       '--dataset-workers') == '14'
    assert final_value(commands['mmmlu_downsampling'],
                       '--dataset-workers') == '14'
    assert final_value(commands['mmlu_prox_downsampling'],
                       '--dataset-workers') == '14'
    assert final_value(commands['mmlu_redux_downsampling'],
                       '--dataset-workers') == '1'
    assert final_value(commands['global_piqa_downsampling'],
                       '--dataset-workers') == '1'
    assert final_value(commands['include_downsampling'],
                       '--dataset-workers') == '14'


def test_core_preflight_retries_after_transient_timeout(tmp_path):
    class DelayedModelsHandler(BaseHTTPRequestHandler):
        attempts = 0

        def do_GET(self):
            type(self).attempts += 1
            if type(self).attempts == 1:
                time.sleep(2)
            body = json.dumps({
                'data': [{
                    'id': 'test-model',
                    'max_model_len': 262144,
                }]
            }).encode()
            try:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), DelayedModelsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = os.environ.copy()
    env.update({
        'PREFLIGHT_TIMEOUT': '1',
        'PREFLIGHT_ATTEMPTS': '2',
        'PREFLIGHT_BACKOFF': '1',
    })
    try:
        result = subprocess.run(
            [
                'bash', str(CORE_SCRIPT), 'ceval_evalscope', '--dry-run',
                '--python', sys.executable, '--model', 'test-model',
                '--base-url', f'http://127.0.0.1:{server.server_port}/v1',
                '--tokenizer-path', 'test/tokenizer', '--work-dir',
                str(tmp_path / 'work'),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            env=env,
            timeout=30,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.returncode == 0, result.stderr
    assert DelayedModelsHandler.attempts == 2
    assert 'preflight attempt 1/2 failed' in result.stderr
    assert 'max_seq_len/max_out_len:' in result.stdout


def test_two_hour_launcher_stages_short_then_full_longbench(tmp_path):
    result = subprocess.run(
        [
            'bash', str(TWO_HOUR_SCRIPT), '--model', 'test-model',
            '--base-url', 'http://127.0.0.1:8000/v1',
            '--tokenizer-path', 'test/tokenizer', '--output-root',
            str(tmp_path), '--run-id', 'test_2h', '--dry-run',
            '--no-metrics-monitor'
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert 'Short stage: aime_2025 ifbench ceval' in result.stdout
    assert 'Long stage: longbench_v2' in result.stdout
    assert result.stdout.index('=== short/aime_2025') < result.stdout.index(
        '=== short/ifbench') < result.stdout.index(
            '=== short/ceval') < result.stdout.index(
                '=== long/longbench_v2')
    child_commands = [line for line in result.stdout.splitlines()
                      if line.startswith('DRY-RUN child:')]
    assert len(child_commands) == 4
    assert all('--max-out-len 32768' in line for line in child_commands)
    assert all('--max-workers 512' in line for line in child_commands[:3])
    assert all('--dataset-workers 1' in line
               for line in child_commands[:2])
    assert '--dataset-workers 32' in child_commands[2]
    assert '--max-workers 32' in child_commands[3]
    assert '--dataset-workers 1' in child_commands[3]
    assert '--query-per-second 8' in child_commands[3]
    assert '--samples all' in child_commands[3]
    assert 'Dry run complete: no inference or scoring was performed.' in result.stdout


def test_two_hour_launcher_marks_long_sampling_as_diagnostic(tmp_path):
    result = subprocess.run(
        [
            'bash', str(TWO_HOUR_SCRIPT), '--model', 'test-model',
            '--base-url', 'http://127.0.0.1:8000/v1', '--output-root',
            str(tmp_path), '--run-id', 'test_2h_sample', '--skip-short',
            '--long-samples', '32', '--dry-run', '--no-metrics-monitor'
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert 'samples=32' in result.stdout
    command = next(line for line in result.stdout.splitlines()
                   if line.startswith('DRY-RUN child:'))
    assert '--samples 32' in command
