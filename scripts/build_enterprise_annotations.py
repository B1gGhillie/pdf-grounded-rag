"""Build reviewable financial QA proposals and page previews from local PDFs.

No proposal becomes an evaluation gold label until a human approves it.
"""
import argparse
import html
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.provenance import atomic_json, file_sha256


def decimal_string(value, places=None):
    value = Decimal(value)
    if places is not None:
        return format(value.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP), "f")
    return format(value, "f")


def number_evidence(page, value, label, year, unit):
    """Require an exact signed source value, with geometry and a nearby excerpt."""
    value = Decimal(value)
    digits = format(abs(value), "f")
    text = page.get_text("text", sort=True)
    compact = re.sub(r"[,，]", "", text)
    token = re.escape(digits)
    expression = (rf"(?<![\d.])(?:-\s*{token}|\(\s*{token}\s*\)|（\s*{token}\s*）)(?![\d.])" if value < 0 else
                  rf"(?<![\d.\-(（]){token}(?![\d.])")
    match = re.search(expression, compact)
    if not match:
        raise ValueError(f"Source value {value} absent on PDF page {page.number+1} ({label})")
    grouped = format(abs(value), ",f")
    boxes = page.search_for(grouped) or page.search_for(digits)
    return {"page": page.number + 1, "label": label, "period": str(year),
            "value": str(value), "unit": unit,
            "bbox_candidates_pt": [[round(v, 2) for v in box] for box in boxes],
            "source_excerpt": compact[max(0, match.start()-100):match.end()+100],
            "note": "Exact signed number found; human must confirm row, column, period and accounting scope."}


def make_document_items(spec, facts, pdf_path):
    company, year = spec["company"], spec["report_year"]
    unit = facts["unit"]
    base = {"doc_id": spec["doc_id"], "company_group": spec["company_group"],
            "company": company, "report_year": year, "split": spec["split"],
            "pdf_path": f"../pdfs/{spec['doc_id']}.pdf", "document_sha256": spec["sha256"],
            "annotation_status": "pending", "reviewer": "", "review_notes": "",
            "full_document_checked": False, "annotation_origin": "assistant_preannotation",
            "evidence_annotation_status": "complete"}
    records = []
    with pymupdf.open(pdf_path) as doc:
        evidence = {}
        descriptors = {
            "revenue": (facts["summary_page"], facts.get("revenue_label", "营业收入"), year),
            "previous_revenue": (facts["summary_page"], facts.get("revenue_label", "营业收入"), year-1),
            "profit": (facts["summary_page"], facts["profit_label"], year),
            "operating_cf": (facts["cf_page"], "经营活动产生的现金流量净额", year),
            "assets": (facts.get("assets_page", facts["summary_page"]), "总资产/资产总计", year),
            "cash": (facts["cash_page"], facts["cash_label"], year),
            "previous_cash": (facts["cash_page"], facts["cash_label"], year-1),
        }
        for key, (page, label, period) in descriptors.items():
            evidence[key] = number_evidence(doc[page-1], facts[key], label, period, unit)

        def add(question, answer, category, keys=(), pages=None, program=None, notes="", answerable=True):
            selected = [evidence[k] for k in keys]
            gold = sorted({e["page"] for e in selected}) if pages is None else pages
            record = {**base, "id": f"cn_{spec['doc_id']}_{len(records)+1:03d}",
                      "question": question, "answer": answer, "answerable": answerable,
                      "gold_pages": gold, "category": category, "evidence": selected,
                      "unit": unit if selected else "", "numeric_program": program,
                      "review_hint": notes or "核对原图中的行名、年份列、单位、负号及证据页；确认后才能批准。",
                      "difficulty": "medium" if category in {"calculation", "cross_page"} else "easy"}
            if not answerable:
                record["evidence_annotation_status"] = "not_applicable"
                record["unanswerable_reason"] = notes
                record["negative_label_basis"] = "hypothesis_only_requires_full_document_review"
            records.append(record)

        add("该报告的报告主体公司全称是什么？", company, "text", pages=[1], notes="核对封面公司全称。")
        add("该报告对应哪一个财务年度？只回答年份。", str(year), "text", pages=[1], notes="核对报告年度，区别发布年份。")
        period_note = facts.get("period_note", "")
        for key, label in [("revenue", facts.get("revenue_label", "营业收入")),
                           ("previous_revenue", facts.get("revenue_label", "营业收入")),
                           ("profit", facts["profit_label"]),
                           ("operating_cf", "经营活动产生的现金流量净额"),
                           ("assets", "年末总资产"), ("cash", "年末"+facts["cash_label"])]:
            period = year-1 if key == "previous_revenue" else year
            scope = "合并口径" if key not in {"profit"} else ""
            add(f"{company}{period}年{scope}{label}是多少？以{unit}回答。" +
                (f"采用调整后口径。" if period_note and key == "previous_revenue" else ""),
                facts[key]+unit, "table", [key], notes=period_note)
        revenue, previous = Decimal(facts["revenue"]), Decimal(facts["previous_revenue"])
        cash, previous_cash = Decimal(facts["cash"]), Decimal(facts["previous_cash"])
        add(f"根据报告中的{year}年和{year-1}年合并收入数值计算收入同比变动率，保留两位小数，下降用负数表示。",
            decimal_string((revenue/previous-1)*100, 2)+"%", "calculation", ["revenue", "previous_revenue"],
            program={"operation": "percent_change", "operands": [str(revenue), str(previous)], "round_digits": 2},
            notes="按原始金额计算，不能直接抄报告中可能只保留一位小数的增长率；核对上年调整后口径。")
        add(f"{year}年合并收入减去{year-1}年合并收入的差额是多少？以{unit}回答，减少用负数表示。",
            decimal_string(revenue-previous)+unit, "calculation", ["revenue", "previous_revenue"],
            program={"operation": "subtract", "operands": [str(revenue), str(previous)]})
        divisor = "100" if unit == "人民币百万元" else "100000000"
        add(f"把{year}年末合并总资产换算为人民币亿元，保留两位小数。",
            decimal_string(Decimal(facts["assets"])/Decimal(divisor), 2)+"亿元", "calculation", ["assets"],
            program={"operation": "divide", "operands": [facts["assets"], divisor], "round_digits": 2})
        if "quarters" in facts:
            for i, amount in enumerate(facts["quarters"]):
                evidence[f"quarter{i+1}"] = number_evidence(doc[facts["quarter_page"]-1], amount,
                    f"第{i+1}季度营业收入", year, unit)
            best = max(range(4), key=lambda i: Decimal(facts["quarters"][i]))
            quarters = [f"quarter{i+1}" for i in range(4)]
            add(f"报告所列{year}年四个季度中，哪一个季度营业收入最高？", f"第{best+1}季度", "table", quarters)
            add(f"{year}年第四季度营业收入比第一季度多多少？以{unit}回答，减少用负数表示。",
                decimal_string(Decimal(facts["quarters"][3])-Decimal(facts["quarters"][0]))+unit,
                "calculation", ["quarter4", "quarter1"],
                program={"operation": "subtract", "operands": [facts["quarters"][3], facts["quarters"][0]]})
        else:
            add(f"{year}年{facts['profit_label']}是正数还是负数？", "负数" if Decimal(facts["profit"])<0 else "正数",
                "table", ["profit"], notes="万科括号表示负数；华为净利润不等同于上市公司归母净利润。")
            add(f"报告所列{year-1}年末{facts['cash_label']}是多少？以{unit}回答。",
                facts["previous_cash"]+unit, "table", ["previous_cash"])
        add(f"分别列出{year}年合并收入和年末{facts['cash_label']}，以{unit}回答。",
            f"收入{facts['revenue']}{unit}；{facts['cash_label']}{facts['cash']}{unit}。",
            "cross_page", ["revenue", "cash"], notes="已提供两处证据；核对是否存在同页替代证据，跨页类型是候选标签。")
        add(f"{year}年末{facts['cash_label']}相较上年末的同比变动率是多少？保留两位小数，下降用负数表示。",
            decimal_string((cash/previous_cash-1)*100, 2)+"%", "calculation", ["cash", "previous_cash"],
            program={"operation": "percent_change", "operands": [str(cash), str(previous_cash)], "round_digits": 2})
        add(f"根据报告，{year}年末{facts['cash_label']}相当于当年合并收入的百分之几？保留两位小数。",
            decimal_string(cash/revenue*100, 2)+"%", "cross_page", ["cash", "revenue"],
            program={"operation": "ratio_percent", "operands": [str(cash), str(revenue)], "round_digits": 2},
            notes="仅为题目指定的两个金额之比，并非流动比率；核对同一合并范围和币种，以及同页替代证据。")
        negative_cases = [
            (f"仅根据这份报告，{company}{year+1}年全年实际合并营业收入是多少？",
             "future_actual", "要求下一财务年度的实际金额，区别预测或目标；须检索全文确认。"),
            (f"仅根据这份报告，{company}{year}年12月单月合并营业收入是多少？请勿用第四季度数据代替。",
             "missing_granularity", "要求单月数据，不能用季度或全年数据替代；须检索全文确认。"),
            (f"仅根据这份报告，{company}{year}年12月单月经营活动现金流量净额是多少？",
             "missing_granularity", "要求单月现金流，区别全年和季度；须检索全文确认。"),
            (f"仅根据这份报告，{'浙江正泰电器股份有限公司' if spec['company_group'] != 'chint' else '深圳市沃尔核材股份有限公司'}{year}年末合并货币资金余额是多少？",
             "different_entity", "问题主体为另一家公司；必须检查全文是否真的未提供，不能只按文件名拒答。"),
        ]
        all_text = "\n".join(p.get_text() for p in doc)
        for question, subtype, note in negative_cases:
            add(question, "", "unanswerable", pages=[], notes=note, answerable=False)
            tokens = [str(year+1)] if subtype == "future_actual" else ["12月", "单月", "营业收入", "现金流"]
            records[-1]["negative_subtype"] = subtype
            records[-1]["search_terms"] = tokens
            records[-1]["search_note"] = "关键词命中/未命中仅用于定位，不证明答案是否存在。"
            records[-1]["keyword_occurrences"] = {token: all_text.count(token) for token in tokens}
        if len(records) != 20:
            raise AssertionError("Each company gets 20 core proposals")
    return records


def chart_items(spec):
    base = {"doc_id": spec["doc_id"], "company_group": spec["company_group"], "company": spec["company"],
            "report_year": 2025, "split": spec["split"], "pdf_path": "../pdfs/huawei_2025.pdf",
            "document_sha256": spec["sha256"], "annotation_status": "pending", "reviewer": "",
            "review_notes": "", "full_document_checked": False, "answerable": True,
            "gold_pages": [7], "category": "chart", "annotation_origin": "assistant_preannotation",
            "evidence_annotation_status": "complete", "chart_with_tabular_duplicate": True,
            "review_hint": "人工看第7页下半部图表，确认系列、横轴年份、单位和答案；上半部有同值表格，不可宣称是纯图表任务。"}
    examples = [
        ("第7页下方的销售收入图中，2021年的销售收入是多少？以人民币百万元回答。", "636807人民币百万元"),
        ("第7页下方的销售收入图中，2021至2025年哪一年的销售收入最高？", "2025年"),
        ("第7页下方的营业利润图中，2021至2025年哪一年的营业利润最高？", "2021年"),
        ("第7页下方的营业利润图中，2022年的营业利润是多少？以人民币百万元回答。", "42216人民币百万元"),
        ("第7页下方的经营活动现金流图中，2021至2025年哪一年的数值最低？", "2022年"),
        ("第7页下方的经营活动现金流图中，2025年的经营活动现金流是多少？以人民币百万元回答。", "127384人民币百万元"),
    ]
    return [{**base, "id": f"cn_huawei_2025_chart_{i+1:03d}", "question": q, "answer": a,
             "evidence": [{"page": 7, "source_region": "lower_half_chart", "requires_visual_review": True}]}
            for i, (q, a) in enumerate(examples)]


def render_previews(root, records):
    poppler = shutil.which("pdftoppm")
    page_sets = {}
    for record in records:
        page_sets.setdefault(record["doc_id"], set()).update(record["gold_pages"])
    for doc_id, pages in page_sets.items():
        out = root/"previews"/doc_id
        out.mkdir(parents=True, exist_ok=True)
        with pymupdf.open(root/"pdfs"/f"{doc_id}.pdf") as doc:
            for page in sorted(pages):
                target = out/f"page_{page:04d}.png"
                if target.exists():
                    continue
                if poppler:
                    subprocess.run([poppler, "-f", str(page), "-l", str(page), "-scale-to", "1500",
                                    "-png", "-singlefile", str(root/"pdfs"/f"{doc_id}.pdf"), str(target.with_suffix(""))],
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                else:
                    doc[page-1].get_pixmap(matrix=pymupdf.Matrix(1.7, 1.7)).save(target)
    return sum(len(p) for p in page_sets.values())


def build(root):
    root = Path(root).resolve()
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    facts = json.loads((root/"seed_facts.json").read_text(encoding="utf-8"))["documents"]
    records = []
    for spec in manifest["documents"]:
        if spec["split"] == "supplementary":
            continue
        records += make_document_items(spec, facts[spec["doc_id"]], root/spec["pdf_path"])
    huawei = next(d for d in manifest["documents"] if d["doc_id"] == "huawei_2025")
    records += chart_items(huawei)
    annotation_root = root/"annotations"
    if any((annotation_root/name).exists() for name in ("dev.json", "test.json")):
        raise ValueError("Existing review files must not be overwritten")
    metadata = {"schema_version": 1, "name": manifest["name"], "annotation_status": "pending",
                "page_numbering": manifest["page_numbering"], "manifest_sha256": file_sha256(root/"manifest.json"),
                "evaluation_note": "Preannotations only. Review all items before exporting a gold dev/test dataset."}
    for split in ("dev", "test"):
        atomic_json(annotation_root/f"{split}.json", {**metadata, "split": split,
                    "items": [r for r in records if r["split"] == split]})
    preview_count = render_previews(root, records)
    report = {"documents": manifest["summary"], "proposals": len(records),
              "answerable_proposals": sum(r["answerable"] for r in records),
              "unanswerable_proposals": sum(not r["answerable"] for r in records),
              "by_split": dict(Counter(r["split"] for r in records)),
              "by_category": dict(Counter(r["category"] for r in records)),
              "human_approved": 0, "previews": preview_count,
              "limitations": ["Nine companies only; convenience sample, not a representative benchmark.",
                              "Chart proposals all from one company/page and also answerable from a duplicate table.",
                              "Cross-page tags are provisional: alternative single-page evidence may exist.",
                              "All negative labels require full-document human review.",
                              "Native extraction can contain font-mapping errors without replacement characters."]}
    atomic_json(root/"annotation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/enterprise_cn")
    build(parser.parse_args().root)
