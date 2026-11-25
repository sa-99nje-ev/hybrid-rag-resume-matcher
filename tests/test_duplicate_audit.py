from scripts.audit_duplicates import audit

def test_duplicate_audit_preserves_kind_and_detects_cross_kind():
    records=[{'id':'a','kind':'resume','text':'C++  Python'}, {'id':'b','kind':'resume','text':'c++ python'}, {'id':'c','kind':'job','text':'C++ Python'}]
    summary,_=audit(records)
    assert summary['exact_groups']==1
    assert summary['exact_duplicate_documents']==1
    assert summary['cross_kind_exact_text_groups']==1

def test_duplicate_audit_reports_prior_id_overlap():
    records=[{'id':'a','kind':'job','text':'Data Engineer'}]
    result,_=audit(records,[{'id':'a','kind':'job','text':'Data Engineer'}])
    assert result['source_order_previous_id_overlap']==1
