"""Create an offline HTML annotation form with embedded dev/test proposals."""
import argparse
import json
from pathlib import Path


TEMPLATE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>中文企业 PDF 问答复核</title>
<style>
:root{font-family:Arial,"Microsoft YaHei",sans-serif;color:#20262b;background:#f7f8fa}*{box-sizing:border-box}body{margin:0}header{padding:18px 24px;border-bottom:1px solid #d7dce1;background:white}h1{font-size:21px;margin:0 0 8px}p{line-height:1.6;margin:6px 0}button,input,select,textarea{font:inherit}button{padding:8px 13px;border:1px solid #b8c1ca;border-radius:4px;background:white;cursor:pointer}button.primary{background:#164f79;color:white;border-color:#164f79}input,select,textarea{padding:8px;border:1px solid #b8c1ca;border-radius:4px}textarea{width:100%;resize:vertical;min-height:64px}label.field{display:block;margin-top:13px;font-size:14px}label.field>input,label.field>select{display:block;width:100%;margin-top:6px}label.field>textarea{margin-top:6px}.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.layout{display:grid;grid-template-columns:250px minmax(340px,1fr) minmax(400px,1.3fr);gap:18px;padding:18px;align-items:start}.panel{background:white;border:1px solid #d7dce1;border-radius:5px;padding:16px}#list{max-height:75vh;overflow:auto}.item{width:100%;text-align:left;margin:4px 0;padding:8px;font-size:13px}.item.selected{border:2px solid #164f79;background:#edf4f8}.muted{color:#5d6872;font-size:13px}.checks{font-size:14px;line-height:1.7;margin:12px 0}.checks label{display:block;margin:8px 0}#preview{max-width:100%;height:auto;display:block;border:1px solid #d7dce1;margin-top:12px}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.6;max-height:220px;overflow:auto;background:#f7f8fa;padding:10px}#message{min-height:23px;color:#943000;font-size:14px;margin:10px 0}#hint{border-left:3px solid #b88425;padding-left:12px;margin:12px 0;font-size:14px}a{color:#164f79}footer{padding:8px 24px 20px;font-size:13px;color:#5d6872}@media(max-width:1100px){.layout{grid-template-columns:200px 1fr}.evidence{grid-column:1/-1}}@media(max-width:650px){.layout{grid-template-columns:1fr}}
</style></head><body>
<header><h1>中文企业 PDF 问答复核</h1>
<p>这是待确认的预标注。对照原 PDF 核对问题、答案、单位、年份和物理页码，再逐题批准。不可回答题需要全文核查。</p>
<div class="controls"><label>数据组 <select id="split"><option value="dev">开发验证集（dev）</option><option value="test">独立测试集（test）</option></select></label><label>复核人 <input id="reviewer" placeholder="姓名或标注者编号"></label><button id="save" class="primary">保存本题</button><button id="download">下载当前组复核 JSON</button><label>恢复 JSON <input type="file" id="upload" accept=".json,application/json"></label></div>
<p class="muted">下载文件保存到项目的 data/enterprise_cn/reviewed/。退出前请下载；刷新会恢复初始预标注。开发集用于调参，测试集在方案冻结后评测。</p>
</header>
<main class="layout"><aside class="panel"><label class="field">筛选状态<select id="filter"><option value="all">全部</option><option value="pending">待复核</option><option value="approved">已批准</option><option value="needs_fix">待修改</option><option value="rejected">已排除</option></select></label><p id="counts" class="muted"></p><div id="list"></div></aside>
<section class="panel"><p id="identity" class="muted"></p><div id="hint"></div>
<label class="field">问题<textarea id="question" rows="4"></textarea></label>
<label class="field">标准答案（数值题保留单位；不可回答留空）<textarea id="answer" rows="3"></textarea></label>
<div class="controls"><label class="field">可回答<select id="answerable"><option value="true">是</option><option value="false">否</option></select></label><label class="field">题型<select id="category"><option value="text">正文</option><option value="table">表格</option><option value="calculation">数值计算</option><option value="cross_page">跨页</option><option value="chart">图表</option><option value="unanswerable">不可回答</option></select></label></div>
<label class="field">证据物理页码（用逗号分隔，例如 7,81）<input id="pages"></label>
<div class="checks"><label><input type="checkbox" id="sourceChecked">我已核对所标证据页的原图、行列、单位和年份</label><label><input type="checkbox" id="fullChecked">我已检查全文，确认文档无法回答此问题</label></div>
<label class="field">复核状态<select id="status"><option value="pending">待复核</option><option value="approved">批准</option><option value="needs_fix">待修改</option><option value="rejected">排除</option></select></label>
<label class="field">复核备注（修订答案、替代证据或排除理由）<textarea id="notes" rows="3"></textarea></label>
<div id="message" role="status"></div><div class="controls"><button id="previous">上一题</button><button id="next">保存并下一题</button></div>
</section>
<section class="panel evidence"><div class="controls"><strong>证据原图</strong><select id="imagePage" aria-label="选择证据页"></select><a id="pdfLink" target="_blank" rel="noopener">打开完整 PDF</a><a id="textLink" target="_blank" rel="noopener">搜索抽取全文</a></div><p id="imageNote" class="muted"></p><img id="preview" alt="证据页预览"><details><summary>候选取值、定位片段与计算程序</summary><pre id="details"></pre></details></section></main>
<footer>样本：9 家企业，10 份 PDF（其中 1 份监管报告单列）。题目高度模板化，6 道图表题均来自华为同页且存在重复表格；适用于预实验，不能据此证明跨企业纯图表泛化。</footer>
<script type="application/json" id="embedded">__DATA__</script><script>
const datasets=JSON.parse(document.getElementById('embedded').textContent);let currentSplit='dev',currentId=datasets.dev.items[0].id;
const el=id=>document.getElementById(id);const current=()=>datasets[currentSplit].items.find(r=>r.id===currentId);
const statusNames={pending:'待复核',approved:'已批准',needs_fix:'待修改',rejected:'已排除'};
function message(text){el('message').textContent=text;}
function renderList(){const rows=datasets[currentSplit].items;const count=state=>rows.filter(r=>r.annotation_status===state).length;el('counts').textContent=`共 ${rows.length} 题 · 批准 ${count('approved')} · 待复核 ${count('pending')} · 待修改 ${count('needs_fix')} · 排除 ${count('rejected')}`;el('list').replaceChildren();for(const r of rows.filter(r=>el('filter').value==='all'||r.annotation_status===el('filter').value)){const button=document.createElement('button');button.className='item'+(r.id===currentId?' selected':'');button.textContent=`${r.company.replace('股份有限公司','')} · ${r.id.split('_').slice(-1)} · ${statusNames[r.annotation_status]}`;button.onclick=()=>{if(saveCurrent()){currentId=r.id;render();}};el('list').append(button);}}
function render(){const r=current();el('identity').textContent=`${r.id} | ${r.company} | ${r.report_year}年度`;el('hint').textContent=r.review_hint;el('question').value=r.question;el('answer').value=r.answer;el('answerable').value=String(r.answerable);el('category').value=r.category;el('pages').value=r.gold_pages.join(',');el('status').value=r.annotation_status;el('notes').value=r.review_notes||'';el('sourceChecked').checked=r.source_pages_checked===true;el('fullChecked').checked=r.full_document_checked===true;el('details').textContent=JSON.stringify({evidence:r.evidence,numeric_program:r.numeric_program,negative_subtype:r.negative_subtype,search_terms:r.search_terms},null,2);el('pdfLink').href=`pdfs/${r.doc_id}.pdf#page=${r.gold_pages[0]||1}`;el('textLink').href=`corpus/${r.doc_id}/pages.txt`;el('imagePage').replaceChildren();for(const p of r.gold_pages){const option=document.createElement('option');option.value=p;option.textContent=`物理第 ${p} 页`;el('imagePage').append(option);}showImage();renderList();message('');}
function showImage(){const r=current(),p=el('imagePage').value;el('preview').hidden=!p;el('imageNote').textContent=p?'物理页码包括封面。可在新窗口打开完整 PDF 放大核对。':'不可回答候选：请打开完整 PDF 和抽取全文进行搜索、核实。';if(p)el('preview').src=`previews/${r.doc_id}/page_${String(p).padStart(4,'0')}.png`;}
el('preview').onerror=()=>{el('preview').hidden=true;el('imageNote').textContent='该修订页尚无预览，请打开完整 PDF 核对。';};
function saveCurrent(){const r=current(),status=el('status').value,answerable=el('answerable').value==='true',reviewer=el('reviewer').value.trim(),answer=el('answer').value.trim(),question=el('question').value.trim(),raw=el('pages').value.trim();let pages=[];try{pages=raw?raw.split(/[,，\s]+/).map(x=>{if(!/^\d+$/.test(x))throw Error('页码必须是正整数');return Number(x);}):[];if(pages.some(p=>p<1)||new Set(pages).size!==pages.length)throw Error('页码不能重复或小于1');if(!question)throw Error('问题不能为空');if(['approved','rejected'].includes(status)&&!reviewer)throw Error('请填写复核人');if(status==='rejected'&&!el('notes').value.trim())throw Error('排除时请填写理由');if(status==='approved'){if(answerable&&(!answer||!pages.length||!el('sourceChecked').checked))throw Error('可回答题需有答案、证据页，并勾选原图核对');if(!answerable&&(answer||pages.length||!el('fullChecked').checked))throw Error('不可回答题需留空答案与证据页，并勾选全文核查');}Object.assign(r,{question,answer,answerable,gold_pages:pages,category:el('category').value,annotation_status:status,reviewer,review_notes:el('notes').value.trim(),source_pages_checked:el('sourceChecked').checked,full_document_checked:el('fullChecked').checked});renderList();message('本题已保存到当前页面；退出前请下载复核 JSON。');return true;}catch(error){message(error.message);return false;}}
function move(delta){if(!saveCurrent())return;const rows=datasets[currentSplit].items,index=rows.findIndex(r=>r.id===currentId);currentId=rows[Math.max(0,Math.min(rows.length-1,index+delta))].id;render();}
el('save').onclick=saveCurrent;el('previous').onclick=()=>move(-1);el('next').onclick=()=>move(1);el('filter').onchange=renderList;el('imagePage').onchange=showImage;
el('split').onchange=()=>{const next=el('split').value;if(!saveCurrent()){el('split').value=currentSplit;return;}currentSplit=next;currentId=datasets[next].items[0].id;el('filter').value='all';render();};
el('download').onclick=()=>{if(!saveCurrent())return;const blob=new Blob([JSON.stringify(datasets[currentSplit],null,2)],{type:'application/json;charset=utf-8'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download=`${currentSplit}.reviewed.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),5000);message('已下载当前组。退出前可继续下载新版；恢复时使用最新 JSON。');};
el('upload').onchange=async()=>{try{const file=el('upload').files[0];if(!file)return;const value=JSON.parse(await file.text()),split=value.split;if(!['dev','test'].includes(split)||!Array.isArray(value.items))throw Error('复核文件格式不正确');const original=datasets[split],expected=new Map(original.items.map(r=>[r.id,r]));if(value.manifest_sha256!==original.manifest_sha256||value.items.length!==expected.size||new Set(value.items.map(r=>r.id)).size!==expected.size)throw Error('复核文件不属于这批资料或题目集合不完整');for(const row of value.items){const old=expected.get(row.id);if(!old||['doc_id','company_group','document_sha256','split'].some(k=>row[k]!==old[k]))throw Error('文档或数据划分发生变化');if(!Object.keys(statusNames).includes(row.annotation_status)||typeof row.answerable!=='boolean'||!Array.isArray(row.gold_pages)||typeof row.question!=='string'||typeof row.answer!=='string')throw Error('题目字段格式不正确');}datasets[split]=value;currentSplit=split;currentId=value.items[0].id;el('split').value=split;el('reviewer').value=value.items.find(r=>r.reviewer)?.reviewer||'';render();message('已恢复复核进度。');}catch(error){message(error.message);}finally{el('upload').value='';}};
render();
</script></body></html>'''


def create(root):
    root = Path(root)
    data = {s: json.loads((root/"annotations"/f"{s}.json").read_text(encoding="utf-8")) for s in ("dev", "test")}
    # Prevent source strings from terminating the embedded script element.
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    target = root/"review.html"
    target.write_text(TEMPLATE.replace("__DATA__", encoded), encoding="utf-8")
    print(target.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/enterprise_cn")
    create(parser.parse_args().root)
