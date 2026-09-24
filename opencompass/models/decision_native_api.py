"""Native candidate probabilities, routed to independent single-GPU replicas."""
from concurrent.futures import ThreadPoolExecutor
import json
from queue import Queue
import urllib.request

from opencompass.registry import MODELS
from .base_api import BaseAPIModel


@MODELS.register_module()
class DecisionNativeAPI(BaseAPIModel):
    def __init__(self,path,endpoints,**kwargs):
        super().__init__(path=path,**kwargs)
        self.endpoints=Queue()
        for endpoint in endpoints:self.endpoints.put(endpoint.rstrip('/'))
        self.workers=len(endpoints)
        self.max_workers=self.workers

    def generate(self,inputs,max_out_len=1,**kwargs):
        def one(value):
            if not isinstance(value,str):
                value=''.join(item['prompt'] for item in value)
            envelope=json.loads(value)
            body={'model':self.path,**envelope['request']}
            endpoint=self.endpoints.get()
            try:
                request=urllib.request.Request(endpoint+'/v1/decision',json.dumps(body).encode(),{'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=300) as response:result=json.load(response)
                if result['model']!=self.path or result['probability_source']!='pointer_head':
                    raise ValueError('Unexpected native model/probability source')
                probs=result['probabilities']
                if envelope['kind']=='jevbench':
                    probs=probs['decision']
                    if envelope['request']['questions']['decision']['type']=='noul':
                        probs={'no':probs['false'],'yes':probs['true']}
                return json.dumps({'probabilities':probs},ensure_ascii=False)
            finally:self.endpoints.put(endpoint)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            return list(pool.map(one,inputs))

    def get_ppl(self,*args,**kwargs):
        raise NotImplementedError('Native pointer head does not expose language-model perplexity')
