from scripts.evaluate_finrank import metrics

def test_binary_metrics_use_k_denominator_and_all_gold():
    m=metrics(['g1','n','g2'],{'g1','g2'})
    assert m['P@5']==0.4
    assert m['R@5']==1.0
    assert m['MRR']==1.0
    assert abs(m['MAP']-(1+2/3)/2)<1e-12
    assert m['nDCG@5']<1.0

def test_incomplete_candidate_ranking_is_rejected():
    try:metrics(['n1','n2'],{'gold'})
    except ValueError:return
    assert False
