"""Conservative deterministic source and arithmetic checks; no semantic entailment claim."""
import ast
import re
import unicodedata
from decimal import Decimal, InvalidOperation, DivisionByZero, localcontext
from .financial_quantities import canonical, compatible


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', str(text)).split())


def quotation_valid(passage_id, quotation, evidence):
    return bool(passage_id in evidence and normalize(quotation) and normalize(quotation) in normalize(evidence[passage_id]))


def decimal_expression(formula, variables):
    """Evaluate a small arithmetic AST using Decimal, excluding code execution."""
    if len(formula) > 512 or len(variables) > 50:
        raise ValueError('formula too large')
    tree = ast.parse(formula, mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError('formula too complex')
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name) and node.id in variables:
            return Decimal(str(variables[node.id]))
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Decimal(ast.get_source_segment(formula, node))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            val = visit(node.operand)
            return val if isinstance(node.op, ast.UAdd) else -val
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            return left / right
        raise ValueError('unsupported formula')
    with localcontext() as ctx:
        ctx.prec = 28
        result = visit(tree)
        if not result.is_finite(): raise ValueError('nonfinite result')
        return result


def expression_dimension(formula, names):
    """Permit only dimensionally interpretable basic financial arithmetic."""
    def walk(node):
        if isinstance(node, ast.Expression): return walk(node.body)
        if isinstance(node, ast.Name) and node.id in names: return names[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int,float): return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):return walk(node.operand)
        if isinstance(node, ast.BinOp):
            left,right=walk(node.left),walk(node.right)
            if isinstance(node.op,(ast.Add,ast.Sub)):
                if left!=right:raise TypeError('incompatible addition')
                return left
            if isinstance(node.op,ast.Mult):
                if left is not None and right is not None:raise TypeError('compound dimensions unsupported')
                return left or right
            if isinstance(node.op,ast.Div):
                if right is None:return left
                if left==right:return None
                raise TypeError('incompatible division')
        raise ValueError('unsupported expression')
    return walk(ast.parse(formula,mode='eval'))

def verify(facts, formula, evidence, question=None):
    if not facts or not formula:
        return {'status':'INSUFFICIENT_EVIDENCE','result':None}
    vars_ = {}
    periods, units = set(), set()
    quantities = []
    for fact in facts:
        q = fact.get('quotation',''); pid = fact.get('passage_id','')
        if not quotation_valid(pid, q, evidence):
            return {'status':'SOURCE_VALUE_UNVERIFIED','result':None}
        value = str(fact.get('value','')).replace(',','').strip()
        try: number = Decimal(value)
        except InvalidOperation: return {'status':'SOURCE_VALUE_UNVERIFIED','result':None}
        if not number.is_finite(): return {'status':'SOURCE_VALUE_UNVERIFIED','result':None}
        # Parse whole numeric tokens, not substrings; support parenthesized negative values.
        nums = re.findall(r'(?<![\w.])\(?-?\d[\d,]*(?:\.\d+)?\)?(?![\w.])', normalize(q))
        def parsed(token):
            token=token.replace(',','')
            try: return -Decimal(token[1:-1]) if token.startswith('(') and token.endswith(')') else Decimal(token)
            except InvalidOperation: return None
        if number not in [parsed(t) for t in nums]:
            return {'status':'SOURCE_VALUE_UNVERIFIED','result':None}
        for field in ('unit','magnitude','reporting_period'):
            val=str(fact.get(field,'')).strip()
            if field=='magnitude' and not val and str(fact.get('unit','')).lower() in ('%','percent','percentage','ratio','times','x'):
                continue
            if not val or normalize(val).lower() not in normalize(q).lower():
                return {'status':'PERIOD_MISMATCH' if field=='reporting_period' else 'UNIT_MISMATCH','result':None}
        currency=str(fact.get('currency','')).strip()
        if currency and currency.lower() not in normalize(q).lower():
            return {'status':'UNIT_MISMATCH','result':None}
        name=fact.get('name','')
        if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',name) or name in vars_:
            return {'status':'INVALID_FORMULA','result':None}
        try: quantity=canonical(fact)
        except ValueError:return {'status':'UNIT_MISMATCH','result':None}
        quantity['name']=name; quantities.append(quantity)
        vars_[name]=number
        units.add((quantity['kind'],quantity['currency']))
        periods.add(str(fact['reporting_period']).lower())
    # Periods differ legitimately in growth calculations; do not assert compatibility automatically.
    if len(periods)>1:
        # Different years are allowed only when all are explicitly requested.
        years=[re.search(r'\b(?:19|20)\d{2}\b', period) for period in periods]
        if not question or any(not y or not re.search(r'(?<!\d)'+re.escape(y.group())+r'(?!\d)',question) for y in years):
            return {'status':'PERIOD_MISMATCH','result':None}
    if len(units)>1: return {'status':'UNIT_MISMATCH','result':None}
    anchor=quantities[0]
    if any(not compatible(anchor,q) for q in quantities):return {'status':'UNIT_MISMATCH','result':None}
    # Normalize explicitly documented magnitudes into the first fact's unit.
    for quantity in quantities:
        vars_[quantity['name']]=quantity['value']*quantity['scale']/anchor['scale']
    try:
        dimension=expression_dimension(formula,{q['name']:q['kind'] for q in quantities})
    except TypeError:return {'status':'UNIT_MISMATCH','result':None}
    except (ValueError,SyntaxError):return {'status':'INVALID_FORMULA','result':None}
    try:
        result=decimal_expression(formula,vars_)
    except (ValueError, SyntaxError, InvalidOperation, ZeroDivisionError, DivisionByZero, OverflowError):
        return {'status':'INVALID_FORMULA','result':None}
    return {'status':'VERIFIED','result':str(result),'evidence_status':'VALID','unit_status':'COMPATIBLE','arithmetic_status':'VALID','answer_status':'NOT_CHECKED','output_dimension':dimension or 'dimensionless','output_scale':str(anchor['scale']),'output_currency':anchor['currency'],'canonical_facts':[{'name':q['name'],'original_value':q['original_value'],'canonical_value':q['canonical_value'],'unit':q['unit'],'currency':q['currency'],'kind':q['kind'],'reporting_period':q['period'],'passage_id':q['passage_id']} for q in quantities], 'note':'Textual source and arithmetic checks do not establish semantic relevance or inferred output unit.'}
