import json
import pytest
from after_training import verify_completion


def test_requires_receipt_and_both_committed_checkpoints(tmp_path):
    loads=[]
    def load(path):
        loads.append(path.name); return {'iteration':72000}
    with pytest.raises(FileNotFoundError): verify_completion(tmp_path,load)
    for name in ('status.json','stop_at_2000_receipt.json'):
        (tmp_path/name).write_text(json.dumps({'additional_iterations':1999}))
    with pytest.raises(ValueError): verify_completion(tmp_path,load)
    for name in ('status.json','stop_at_2000_receipt.json'):
        (tmp_path/name).write_text(json.dumps({'additional_iterations':2000}))
    verify_completion(tmp_path,load)
    assert loads==['checkpoint.pt','additional_2000.pt']
    with pytest.raises(ValueError): verify_completion(tmp_path,lambda p:{'iteration':72001})
