"""Generate clearly synthetic Chinese native/scanned PDF regression fixtures.

Numbers and prose are invented software test inputs, never research results.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pymupdf


CASES = [
    ("industry_native", "行业节能评估回归样例", [
        "本报告为虚构软件测试材料。改造项目年节电量为120万千瓦时。核算口径为设备运行阶段。",
        "项目总投资为350万元。静态回收期为3年。统计范围不包含办公楼。",
        "本报告未披露项目的贷款利率。设备保养周期为6个月。"], [
        ("项目年节电量是多少？", "改造项目年节电量为120万千瓦时。", [1], "numeric_unit"),
        ("统计范围是否包含办公楼？", "统计范围不包含办公楼。", [2], "negation"),
        ("项目的贷款利率是多少？", "", [], "near_domain_unanswerable")]),
    ("thesis_native", "学位论文实验回归样例", [
        "本页为虚构论文测试材料。模型的验证集准确率为91.5%。数据按文档划分以避免信息泄漏。",
        "训练阶段的批量大小为16。学习率为0.001。实验仅使用公开数据。",
        "本论文未提供生产部署的延迟测量结果。该模型不支持实时语音识别。"], [
        ("训练阶段的批量大小是多少？", "训练阶段的批量大小为16。", [2], "numeric"),
        ("该模型是否支持实时语音识别？", "该模型不支持实时语音识别。", [3], "negation"),
        ("生产部署的延迟是多少？", "", [], "near_domain_unanswerable")]),
    ("industry_scan", "行业用水扫描回归样例", [
        "这是虚构的行业用水测试报告。生产线年节水量为8万吨。计量范围仅包括生产车间。",
        "设备采购费用为240万元。安装费用为30万元。验收记录由技术部门保存。",
        "报告未提供污水处理的单位成本。所有金额均为不含税金额。"], [
        ("设备采购费用是多少？", "设备采购费用为240万元。", [2], "scanned_numeric_unit"),
        ("安装费用是多少？", "安装费用为30万元。", [2], "scanned_numeric_unit"),
        ("污水处理的单位成本是多少？", "", [], "near_domain_unanswerable")]),
    ("thesis_scan", "学位论文扫描回归样例", [
        "这是虚构的学位论文扫描测试材料。测试样本数量为500个。评价指标包括准确率和召回率。",
        "实验模型的参数数量为120万个。推理设备仅使用中央处理器。",
        "该论文未报告模型的功耗。后续研究计划增加更多领域文档。"], [
        ("测试样本数量是多少？", "测试样本数量为500个。", [1], "scanned_numeric"),
        ("推理设备使用什么处理器？", "推理设备仅使用中央处理器。", [2], "scanned_factual"),
        ("实验模型的功耗是多少？", "", [], "near_domain_unanswerable")]),
]


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "data/fixtures/chinese"
    out.mkdir(parents=True, exist_ok=True)
    items = []
    for name, title, paragraphs, questions in CASES:
        with pymupdf.open() as doc:
            for paragraph in paragraphs:
                page = doc.new_page()
                page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), title + "\n\n" + paragraph,
                                    fontname="china-s", fontsize=16)
            if "scan" in name:
                with pymupdf.open() as scan:
                    for page in doc:
                        image = page.get_pixmap(dpi=180).tobytes("png")
                        scan.new_page().insert_image(page.rect, stream=image)
                    scan.save(out / f"{name}.pdf")
            else:
                doc.save(out / f"{name}.pdf")
        for n, (question, answer, pages, category) in enumerate(questions, 1):
            items.append({"id": f"{name}_{n}", "pdf_path": f"{name}.pdf", "question": question,
                          "answer": answer, "answerable": bool(answer), "gold_pages": pages,
                          "category": category, "synthetic": True})
    (out / "dataset.json").write_text(json.dumps({"name": "synthetic_chinese_regression",
        "warning": "虚构回归测试数据，不可用作论文效果或新颖性证据", "items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out / "dataset.json")


if __name__ == "__main__":
    main()
