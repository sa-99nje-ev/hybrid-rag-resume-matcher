"""Explicit financial quantity representation and conservative answer comparison."""
import re
from decimal import Decimal,InvalidOperation

SCALES={'':Decimal(1),'unit':Decimal(1),'units':Decimal(1),'one':Decimal(1),'thousand':Decimal(1000),'thousands':Decimal(1000),'million':Decimal(1000000),'millions':Decimal(1000000),'billion':Decimal(1000000000),'billions':Decimal(1000000000)}

def canonical(fact):
    raw=str(fact.get('value','')).replace(',','').strip()
    if not re.fullmatch(r'-?\d+(?:\.\d+)?',raw):raise ValueError('invalid number')
    value=Decimal(raw);magnitude=str(fact.get('magnitude','')).strip().lower()
    if magnitude not in SCALES:raise ValueError('unknown magnitude')
    unit=str(fact.get('unit','')).strip().lower();currency=str(fact.get('currency','')).strip().upper()
    if unit in ('%','percent','percentage'):kind='percentage'
    elif unit in ('ratio','times','x'):kind='ratio'
    elif unit in ('dollar','dollars','usd','currency') and currency:kind='money'
    else:kind='unit:'+unit
    if kind in ('percentage','ratio') and (currency or SCALES[magnitude]!=1):raise ValueError('incompatible quantity')
    return {'original_value':raw,'canonical_value':str(value*SCALES[magnitude]),'value':value,'scale':SCALES[magnitude],
            'unit':unit,'currency':currency,'kind':kind,'period':str(fact.get('reporting_period','')).strip(),
            'passage_id':fact.get('passage_id',''),'quotation':fact.get('quotation','')}

def compatible(a,b):
    return a['kind']==b['kind'] and a['currency']==b['currency']

def answer_agreement(answer,computed,absolute_tolerance=Decimal('0.01'),relative_tolerance=Decimal('0.001')):
    """One unambiguous numeric answer, within max(0.01, 0.1% of result)."""
    found=re.findall(r'(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w.])',answer)
    if len(found)!=1:return False
    try:actual=Decimal(found[0].replace(',',''));expected=Decimal(str(computed))
    except InvalidOperation:return False
    return abs(actual-expected)<=max(absolute_tolerance,abs(expected)*relative_tolerance)
