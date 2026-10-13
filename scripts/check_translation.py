#!/usr/bin/env python3
"""Check source coverage, heading levels, exact fenced code, and local web links."""
import sys,re,json
from pathlib import Path
from urllib.parse import urlsplit,unquote
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import build
R=build.ROOT
fences=lambda s:re.findall(r'```[^\n]*\n.*?```',s,re.S)
clean=lambda s:re.sub(r'```.*?```','',s,flags=re.S)
urls=lambda s:set(re.findall(r'\]\(([^\s)]+)',clean(s)))
expected={str(p.relative_to(build.SRC)) for g in ('training','insights') for p in (build.SRC/g).rglob('*.md')}
assert expected==set(build.ORDER),'Document coverage mismatch'
report={'documents':len(expected),'headings':0,'fenced_code_blocks':0,'errors':[]}
for p in build.ORDER:
 s=(build.SRC/p).read_text();t=(R/'content'/p).read_text()
 if fences(s)!=fences(t):report['errors'].append([p,'code mismatch'])
 if re.findall(r'^(#{1,6})\s',clean(s),re.M)!=re.findall(r'^(#{1,6})\s',clean(t),re.M):report['errors'].append([p,'heading levels mismatch'])
 if urls(s)-urls(t):report['errors'].append([p,'missing original URLs',sorted(urls(s)-urls(t))])
 report['headings']+=len(build.headings(s));report['fenced_code_blocks']+=len(fences(s))
 for u in urls(build.rewrite(t,p)):
  x=urlsplit(u)
  if x.scheme or x.netloc:continue
  target=build.WEB/'chapters'/unquote(x.path) if x.path else build.WEB/'chapters'/(build.IDS[p]+'.html')
  if x.path and not target.exists():report['errors'].append([p,'missing local target',u])
  if x.fragment:
   q=next((k for k,v in build.IDS.items() if v+'.html'==target.name),None)
   if q and unquote(x.fragment) not in build.anchors((build.SRC/q).read_text()):report['errors'].append([p,'missing anchor',u])
(R/'translation-check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
sys.exit(bool(report['errors']))
