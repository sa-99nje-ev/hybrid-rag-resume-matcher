"""Local conservative pacing; explicit organization limits override observed defaults."""
import os,time,re,collections,json
from pathlib import Path

def duration(value):
    if value is None:return None
    try:return max(0.0,float(value))
    except (TypeError,ValueError):
        parts=re.findall(r'(\d+(?:\.\d+)?)(ms|h|m|s)',str(value))
        if not parts:return None
        return sum(float(n)*{'ms':.001,'s':1,'m':60,'h':3600}[unit] for n,unit in parts)

def estimate_tokens(text):
    # Conservative byte-based proxy; provider-reported usage is authoritative.
    return (len(text.encode('utf8'))+2)//3

class ProviderDeferred(Exception):
    pass

class TokenLimiter:
    def __init__(self,clock=time.monotonic,sleep=time.sleep):
        self.clock=clock;self.sleep=sleep;self.events=collections.deque();self.daily=collections.deque();self.observed={};self.remaining_tokens=None;self.tokens_reset_at=0
        self.tpm=int(os.getenv('GROQ_TPM','7500'));self.rpm=int(os.getenv('GROQ_RPM','20'));self.rpd=int(os.getenv('GROQ_RPD','1000'))
        self.itpm=int(os.getenv('GROQ_ITPM','0'));self.otpm=int(os.getenv('GROQ_OTPM','0'))
        self.cooldown_file=Path(__file__).resolve().parents[2]/'data/cache/groq_rate_limit_cooldown.json'
    def set_cooldown(self,seconds):
        self.cooldown_file.parent.mkdir(parents=True,exist_ok=True)
        current=0
        if self.cooldown_file.exists():
            try:current=float(json.loads(self.cooldown_file.read_text(encoding='utf8')).get('not_before',0))
            except (ValueError,OSError):pass
        self.cooldown_file.write_text(json.dumps({'not_before':max(current,time.time()+seconds)}),encoding='utf8')
    def check_cooldown(self):
        if self.cooldown_file.exists():
            try:delay=float(json.loads(self.cooldown_file.read_text(encoding='utf8')).get('not_before',0))-time.time()
            except (ValueError,OSError):raise ProviderDeferred('Unable to read provider cooldown')
            if delay>0:raise ProviderDeferred('Provider cooldown remains active')
    def wait(self,input_estimate,output_reserve):
        self.check_cooldown()
        if self.remaining_tokens is not None and self.remaining_tokens<input_estimate+output_reserve:
            remaining_wait=self.tokens_reset_at-self.clock()
            if remaining_wait>0:self.sleep(remaining_wait+.1)
            self.remaining_tokens=None
        if input_estimate+output_reserve>self.tpm:raise ValueError('estimated request exceeds local token budget')
        while True:
            now=self.clock()
            while self.events and now-self.events[0][0]>=60:self.events.popleft()
            while self.daily and now-self.daily[0]>=86400:self.daily.popleft()
            total=sum(x[1]+x[2] for x in self.events);ins=sum(x[1] for x in self.events);outs=sum(x[2] for x in self.events)
            over=(len(self.events)>=self.rpm or total+input_estimate+output_reserve>self.tpm or (self.itpm and ins+input_estimate>self.itpm) or (self.otpm and outs+output_reserve>self.otpm))
            if len(self.daily)>=self.rpd:raise ValueError('local daily request budget exhausted')
            if not over:break
            if not self.events:raise ValueError('estimated request exceeds configured token budget')
            self.sleep(max(.01,60-(now-self.events[0][0])))
        stamp=self.clock();self.events.append((stamp,input_estimate,output_reserve));self.daily.append(stamp)
    def observe(self,headers):
        allowed=('x-ratelimit-limit-requests','x-ratelimit-limit-tokens','x-ratelimit-remaining-requests','x-ratelimit-remaining-tokens','x-ratelimit-reset-requests','x-ratelimit-reset-tokens','retry-after')
        self.observed={k:str(headers[k]) for k in allowed if k in headers}
        try:self.tpm=min(self.tpm,int(headers.get('x-ratelimit-limit-tokens',self.tpm)))
        except (TypeError,ValueError):pass
        try:self.rpd=min(self.rpd,int(headers.get('x-ratelimit-limit-requests',self.rpd)))
        except (TypeError,ValueError):pass
        try:
            if int(headers.get('x-ratelimit-remaining-requests','1'))<=0:raise ValueError('provider daily request limit reached')
        except TypeError:pass
        try:remaining=int(headers.get('x-ratelimit-remaining-tokens','-1'))
        except ValueError:remaining=-1
        reset=duration(headers.get('x-ratelimit-reset-tokens'))
        if remaining>=0 and reset is not None:
            self.remaining_tokens=remaining;self.tokens_reset_at=self.clock()+reset
