#!/usr/bin/env python3
"""Build the Chinese reading site with pinned source links and original anchors."""
from pathlib import Path
import re,json,html,posixpath,shutil
from urllib.parse import urlsplit,unquote
ROOT=Path(__file__).resolve().parent
SRC=ROOT/'src/ml-engineering'; WEB=ROOT/'webapp'
manifest=json.loads((ROOT/'source-manifest.json').read_text())
REV=manifest['revision']; UP=f'https://github.com/stas00/ml-engineering/blob/{REV}/'
ORDER=['training/README.md','training/model-parallelism/README.md','training/performance/README.md','training/fault-tolerance/README.md','training/reproducibility/README.md','training/instabilities/README.md','training/instabilities/training-loss-patterns.md','training/checkpoints/README.md','training/hparams.md','training/dtype.md','training/emulate-multi-node.md','training/re-train-hub-models.md','training/datasets.md','insights/ai-battlefield.md','insights/how-to-choose-cloud-provider.md','insights/when-to-upgrade-gpus/README.md']
TITLES=['训练导读','模型并行','训练性能优化','容错','可复现性','训练不稳定性','训练损失曲线的形态','检查点','训练超参数与模型初始化','张量精度与数据类型','单节点模拟多节点','从头训练 HF Hub 模型','数据集处理','AI 战场上的工程经验','如何选择云服务商','何时升级 GPU']
IDS={p:p[:-3].replace('/','--') for p in ORDER}
READY=[p for p in ORDER if (ROOT/'content'/p).exists()]
FOOT='非官方中文译本 · 原作者 Stas Bekman 及贡献者 · 原文与译文采用 CC BY-SA 4.0 · 译文保留原文时点与作者观点。'
def headings(md):
 clean=re.sub(r'^(`{3,}|~{3,}).*?^\1[^\n]*$', '', md, flags=re.M|re.S)
 return re.findall(r'^#{1,6}\s+(.+?)\s*#*$',clean,re.M)
def slug(s):
 s=re.sub(r'!?(?:\[([^]]+)\])\([^)]*\)',r'\1',s)
 s=re.sub('<[^>]+>','',s).lower().replace('`','')
 return re.sub(r'[^\w\- ]','',s).replace(' ','-')
def anchors(md):
 counts={};out=[]
 for h in headings(md):
  a=slug(h); n=counts.get(a,0);counts[a]=n+1;out.append(a+(f'-{n}' if n else ''))
 return out
def resolve(url,source):
 u=urlsplit(url)
 if u.scheme or u.netloc or not u.path:return url
 p=posixpath.normpath(posixpath.join(posixpath.dirname(source),unquote(u.path)))
 target=p if p in IDS else p.rstrip('/')+'/README.md'
 if target in READY:return IDS[target]+'.html'+('?' +u.query if u.query else '')+('#'+u.fragment if u.fragment else '')
 if (SRC/p).is_file() and not p.endswith('.md'):
  dest=WEB/'source-assets'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SRC/p,dest)
  return '../source-assets/'+p+('#'+u.fragment if u.fragment else '')
 return UP+p+('?' +u.query if u.query else '')+('#'+u.fragment if u.fragment else '')
def rewrite(md,path):
 # Protect code from link rewriting.
 pieces=re.split(r'(```.*?```)',md,flags=re.S)
 for i in range(0,len(pieces),2):
  pieces[i]=re.sub(r'(!?\[[^\n]*?\]\()([^\s)]+)([^)]*\))',lambda m:m[1]+resolve(m[2],path)+m[3],pieces[i])
  pieces[i]=re.sub(r'((?:src|href)=[\"\x27])([^\"\x27]+)([\"\x27])',lambda m:m[1]+resolve(m[2],path)+m[3],pieces[i])
 return ''.join(pieces)
def page(title,body,base=''):
 return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} · ML Engineering 中文版</title><link rel="stylesheet" href="{base}assets/style.css"><link rel="stylesheet" href="{base}assets/katex/katex.min.css"></head><body>{body}<script src="{base}assets/marked.min.js"></script><script src="{base}assets/katex/katex.min.js"></script><script src="{base}assets/katex/auto-render.min.js"></script><script src="{base}assets/reader.js"></script></body></html>'''
def payload(md,original):
 return '<script type="application/json" class="document-data">'+json.dumps({'markdown':md,'anchors':anchors(original)},ensure_ascii=False).replace('<','\\u003c')+'</script>'
def build():
 (WEB/'chapters').mkdir(parents=True,exist_ok=True)
 full=[]
 for path in READY:
  i=ORDER.index(path);title=TITLES[i];idx=READY.index(path)
  md=rewrite((ROOT/'content'/path).read_text(),path);original=(SRC/path).read_text()
  if len(headings(md))!=len(headings(original)):print('HEADING MISMATCH',path,len(headings(md)),len(headings(original)))
  nav=''
  for ni,label in [(idx-1,'← 上一章'),(idx+1,'下一章 →')]:
   if 0<=ni<len(READY):nav+=f'<a href="{IDS[READY[ni]]}.html">{label} · {TITLES[ORDER.index(READY[ni])]}</a>'
  body=f'''<header class="topbar"><a class="brand" href="../index.html">ML Engineering 中文版</a><nav class="chapnav"><a href="../full.html">全文阅读</a><a href="{UP+path}">英文原文</a></nav></header><div class="layout"><aside class="sidebar"><nav class="toc"><h2>本章目录</h2></nav></aside><main class="main"><article class="prose">{payload(md,original)}</article><nav class="pager">{nav}</nav><footer class="pagefoot">{FOOT}</footer></main></div>'''
  (WEB/'chapters'/f'{IDS[path]}.html').write_text(page(title,body,'../'))
  # In the combined page, chapter links still go to chapter pages for unambiguous anchors.
  fullmd=md.replace('../source-assets/','source-assets/')
  fullmd=re.sub(r'(?<![/\w])((?:training|insights)--[^\s\"<>)]*\.html)',r'chapters/\1',fullmd)
  full.append(f'<section class="full-chapter"><a href="chapters/{IDS[path]}.html">单章阅读 · {html.escape(title)}</a><article class="prose">{payload(fullmd,original)}</article></section>')
 cards=[]
 for group,label in [('training','训练 Training'),('insights','洞见 Insights')]:
  groupcards=[]
  for i,path in enumerate(ORDER):
   if not path.startswith(group):continue
   en=headings((SRC/path).read_text())[0]
   inner=f'<span class="no">{i+1:02}</span><span class="zh">{TITLES[i]}</span><span class="en">{html.escape(en)}</span>'
   groupcards.append(f'<a class="card" href="chapters/{IDS[path]}.html">{inner}</a>' if path in READY else f'<div class="card disabled">{inner}<span class="desc">翻译中</span></div>')
  cards.append('<h2>'+label+'</h2><div class="chapter-grid">'+''.join(groupcards)+'</div>')
 body=f'''<section class="hero"><div class="eyebrow">简体中文 · 非官方翻译</div><h1>机器学习工程实战<br>Training &amp; Insights</h1><p class="sub">Stas Bekman《Machine Learning Engineering Open Book》的训练与洞见两部分。保留原文结构、代码、公式、图表与参考链接。</p><div class="meta"><div><b>{len(READY)} / {len(ORDER)}</b>已译文章</div><div><b>{REV[:8]}</b>原文版本</div><a href="full.html"><b>全文阅读 →</b>连续浏览</a><a href="https://github.com/stas00/ml-engineering"><b>原作 ↗</b>Stas Bekman</a></div></section><div class="grid-wrap">{''.join(cards)}</div><footer class="foot">{FOOT}</footer>'''
 (WEB/'index.html').write_text(page('机器学习工程实战',body))
 (WEB/'full.html').write_text(page('全文阅读',f'<header class="topbar"><a class="brand" href="index.html">ML Engineering 中文版</a></header><main class="full-main">'+''.join(full)+f'<footer class="pagefoot">{FOOT}</footer></main>'))
 print(f'Built {len(READY)}/{len(ORDER)} chapters + index + full')
if __name__=='__main__':build()
