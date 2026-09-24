import json
import pytest
from opencompass.datasets.decision_benchmarks import JevBenchEvaluator, KevEvaluator, parse_json
from opencompass.datasets._decision_upstream.jevbench.scoring import score_task
from opencompass.datasets._decision_upstream.jevbench.tasks import Task


def ref(expected='yes'):
    return dict(id='x', family='policy', state='Some evidence',
                question={'type': 'noul', 'instructions': 'Does it hold?'},
                labels=['no', 'yes'], expected=expected, split='public')


def test_jev_rounding_ties_and_invalid_match_native():
    for p in [{'no': .499, 'yes': .499}, {'no': .2, 'yes': .8},
              {'no': True, 'yes': 0}, {'no': .9}, {'no': .1, 'yes': float('nan')}]:
        native = score_task(p, Task.from_dict(ref()))
        out = JevBenchEvaluator().score([json.dumps({'probabilities': p})], [json.dumps(ref())])
        assert out['accuracy'] == 100 * bool(native['correct'])
        assert out['schema_validity'] == 100 * native['valid']
        if not native['valid']:
            assert out['brier'] is None and out['calibration_n'] == 0


def test_last_probability_object_and_no_label_imputation():
    assert parse_json('thought {"other": 3}\n{"probabilities":{"x":1}}')['probabilities']=={'x':1}
    with pytest.raises(ValueError): parse_json('yes')


def test_kev_invalid_is_not_valid_only_accuracy():
    record={'state':'A', '_meta':{'id':'x','group_id':'x','source':'fixture','variant':'clean'},
            'questions':{'q':{'type':'choice','instructions':'choose','criteria':{'a':None,'b':None},'label':'a','src':'fixture'}}}
    out=KevEvaluator().score(['{"probabilities":{"q":{"a":1}}}'],[json.dumps(record)])
    assert out['invalid_records']==1 and out['accuracy'] is None
    out=KevEvaluator().score(['{"probabilities":{"q":{"a":0.75,"b":0.25}}}'],[json.dumps(record)])
    assert out['accuracy']==100 and out['brier']==pytest.approx(.125)


def test_population_mismatch_rejected():
    with pytest.raises(ValueError): JevBenchEvaluator().score([], [json.dumps(ref())])


def test_jev_request_schema_matches_upstream_and_is_thread_local(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from opencompass.models.decision_api import JevBenchOpenAIAPI
    from opencompass.models.vllm_openai_api import VLLMOpenAIAPI
    from opencompass.datasets._decision_upstream.jevbench.adapters.openai_compat import OpenAICompatAdapter
    model = object.__new__(JevBenchOpenAIAPI)
    model.generation_endpoint = 'chat'
    model.max_seq_len = 8192
    model.mode = 'none'
    model.get_token_len = lambda _: 1
    model._preprocess_messages = lambda x, n, *args: (x, n)
    model.openai_extra_kwargs = {'seed': 7}
    monkeypatch.setattr(VLLMOpenAIAPI, '_generate',
                        lambda self, *args: self.openai_extra_kwargs)
    tasks = [Task.from_dict(ref()), Task.from_dict(dict(ref(), labels=['a', 'b'], expected='a',
             question={'type': 'choice', 'instructions': 'Choose'})),
             Task.from_dict(dict(ref(), labels=['0', '1'], expected=1,
             question={'type': 'score', 'instructions': 'Rate', 'criteria': ['bad', 'good']}))]
    bodies = [OpenAICompatAdapter('', '').build_request(t) for t in tasks]
    with ThreadPoolExecutor(3) as pool:
        out = list(pool.map(lambda b: model._generate(b['messages'], 4096, 0), bodies))
    for result, body in zip(out, bodies):
        assert result['response_format'] == body['response_format']
        assert result['seed'] == 7
    assert model.openai_extra_kwargs == {'seed': 7}


def test_transport_failure_is_not_a_scored_empty_answer():
    from opencompass.models.decision_api import DecisionOpenAIAPI
    model = object.__new__(DecisionOpenAIAPI)
    model.temperature = 0
    model.max_workers = 2
    def fail(*args):
        raise ConnectionError('service unavailable')
    model._generate = fail
    with pytest.raises(ConnectionError):
        model.generate(['prompt'])
