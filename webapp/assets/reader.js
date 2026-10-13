/* Local Markdown + math renderer. Preserve the English heading anchors. */
(() => {
 const allHeads=[];
 document.querySelectorAll('.document-data').forEach((el,docIndex)=>{
  const data=JSON.parse(el.textContent),math=[];
  const prefix=document.querySelector('.full-main')?'doc'+docIndex+'-':'';
  const notes=[];
  const prose=data.markdown.replace(/^\[\^([^\]]+)\]: (.*)$/gm,(_,id,text)=>{notes.push([id,text]);return '';});
  const withNotes=prose.replace(/\[\^([^\]]+)\]/g,(match,id)=>notes.some(note=>note[0]===id)?'<sup><a href="#fn-'+id+'">['+id+']</a></sup>':match)+(notes.length?'\n\n<hr>\n\n'+notes.map(([id,text])=>'<div id="fn-'+id+'">'+marked.parse(text)+'</div>').join('\n'):'');
  const parts=withNotes.split(/(```[\s\S]*?```|`[^`\n]*`)/g);
  const stash=m=>{math.push(m);return 'MATHPLACEHOLDER'+(math.length-1)+'ENDMATH';};
  const md=parts.map((s,i)=>i%2?s:s.replace(/\$\$[\s\S]*?\$\$/g,stash).replace(/\$(?!\s)(?:\\.|[^$\n\\])+?(?<!\s)\$/g,stash)).join('');
  const article=el.parentElement;
  article.innerHTML=marked.parse(md).replace(/MATHPLACEHOLDER(\d+)ENDMATH/g,(_,i)=>math[+i].replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;'));
  article.querySelectorAll('h1,h2,h3,h4,h5,h6').forEach((h,i)=>{
   h.id=(document.querySelector('.full-main')?'doc'+docIndex+'-':'')+(data.anchors[i]||'section-'+i);
   if(h.tagName==='H2'||h.tagName==='H3')allHeads.push(h);
  });
  if(prefix){
   article.querySelectorAll('[id]').forEach(node=>{if(!/^H[1-6]$/.test(node.tagName))node.id=prefix+node.id;});
   article.querySelectorAll('a[href^="#"]').forEach(a=>a.setAttribute('href','#'+prefix+a.getAttribute('href').slice(1)));
  }
  if(window.renderMathInElement)renderMathInElement(article,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false},{left:'\\[',right:'\\]',display:true},{left:'\\(',right:'\\)',display:false}],throwOnError:false});
 });
 const toc=document.querySelector('.toc');
 if(toc){const ol=document.createElement('ol');allHeads.forEach(h=>{const li=document.createElement('li');li.className='toc-'+h.tagName.toLowerCase();const a=document.createElement('a');a.href='#'+h.id;a.textContent=h.textContent;li.append(a);ol.append(li);});toc.append(ol);}
 if(location.hash){const target=document.getElementById(decodeURIComponent(location.hash.slice(1)));if(target)requestAnimationFrame(()=>target.scrollIntoView());}
})();
