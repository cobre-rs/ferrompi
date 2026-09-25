import os,re,sys
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'..','..','..','..'))
rs=open(os.path.join(ROOT,'src','ffi.rs')).read()
c=open(os.path.join(ROOT,'csrc','ferrompi.c')).read()
# strip comments
c=re.sub(r'/\*.*?\*/','',c,flags=re.S); c=re.sub(r'//[^\n]*','',c)
rs=re.sub(r'//[^\n]*','',rs)
def norm_rs(t):
    t=t.strip()
    if t.startswith('*'): return 'ptr'
    m={'int32_t':'i32','c_int':'i32','i32':'i32','int64_t':'i64','i64':'i64','c_double':'f64','f64':'f64','u64':'u64','usize':'usize'}
    return m.get(t,t)
def norm_c(t):
    t=t.replace('const','').strip()
    if '*' in t: return 'ptr'
    t=' '.join(t.split()[:-1]) if len(t.split())>1 else t
    m={'int32_t':'i32','int':'i32','int64_t':'i64','double':'f64','void':'()','uint64_t':'u64','size_t':'usize'}
    return m.get(t.strip(),t.strip())
rsf={}
for m in re.finditer(r'pub fn (\w+)\s*\((.*?)\)\s*(->\s*([\w\s\*]+))?;',rs,re.S):
    name=m.group(1); args=[a for a in m.group(2).split(',') if a.strip()]
    types=[norm_rs(a.split(':',1)[1]) for a in args]
    ret=norm_rs(m.group(4)) if m.group(4) else '()'
    rsf[name]=(types,ret)
cf={}
for m in re.finditer(r'\n(int32_t|int64_t|int|double|void)\s+(ferrompi_\w+)\s*\(([^)]*)\)\s*\{',c,re.S):
    ret,name,args=m.group(1),m.group(2),m.group(3)
    a=[x for x in args.split(',') if x.strip() and x.strip()!='void']
    types=[norm_c(x) for x in a]
    retn={'int32_t':'i32','int':'i32','int64_t':'i64','double':'f64','void':'()'}[ret]
    cf.setdefault(name,[]).append((types,retn))
bad=0
for n,(t,r) in rsf.items():
    if n not in cf: print('MISSING in C:',n); bad+=1; continue
    ok=any(t==ct and r==cr for ct,cr in cf[n])
    if not ok: print('MISMATCH',n,'rs',t,r,'c',cf[n]); bad+=1
print('checked',len(rsf),'bad',bad)
