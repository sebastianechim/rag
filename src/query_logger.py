import os, json, time
from typing import List, Dict, Any, Optional

LOCAL_LOG_DIR = os.environ.get('LOCAL_LOG_DIR','logs')
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)

def _write_local(record: Dict[str, Any]):
    fname = f"{LOCAL_LOG_DIR}/queries_{time.strftime('%Y%m%d')}.ndjson"
    with open(fname,'a') as f:
        f.write(json.dumps(record, default=str)+'\n')

def log_query(query_text: str, retrieved: List[Dict[str,Any]], answer_text: str, latency_ms: float, user_id: Optional[str]=None):
    ts = int(time.time())
    rec = {'timestamp':ts,'query':query_text,'user_id':user_id,'retrieved_ids':[r.get('chunk_id') for r in retrieved],'retrieved_meta':retrieved,'answer':answer_text,'latency_ms':latency_ms}
    _write_local(rec)
