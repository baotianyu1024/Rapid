"""Render a reviewer response Markdown file as a print-ready PDF."""

import argparse
import sys
from pathlib import Path

import markdown


sys.path.insert(0, '/tmp/rapid_response_pdf_deps')
from weasyprint import HTML  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('source', nargs='?', default='中文Response.md')
parser.add_argument('target', nargs='?')
args = parser.parse_args()
source = (ROOT / args.source).resolve()
target = (ROOT / args.target).resolve() if args.target else source.with_suffix('.pdf')
body = markdown.markdown(
    source.read_text(encoding='utf-8'),
    extensions=['tables', 'fenced_code', 'sane_lists', 'toc'],
)
html = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><style>
@font-face {{ font-family: "Response CJK";
  src: url("file:///usr/share/fonts/google-droid/DroidSansFallback.ttf");
  font-weight: normal; }}
@font-face {{ font-family: "Response CJK";
  src: url("file:///usr/share/fonts/google-droid/DroidSansFallback.ttf");
  font-weight: bold; }}
@page {{ size: A4; margin: 18mm 16mm 17mm;
  @bottom-right {{ content: counter(page); font: 8pt "Response CJK"; color: #687480; }} }}
body {{ font: 9.3pt/1.52 "Response CJK", sans-serif; color: #1e2933; }}
h1 {{ font-size: 19pt; line-height: 1.18; color: #162b3b;
  border-bottom: 2px solid #2b6176; padding-bottom: 8pt; margin: 0 0 16pt; }}
h2 {{ font-size: 13pt; line-height: 1.3; color: #174c60;
  margin: 19pt 0 7pt; break-after: avoid; }}
h3 {{ font-size: 10.5pt; margin: 13pt 0 5pt; break-after: avoid; }}
p {{ margin: 0 0 8pt; orphans: 2; widows: 2; }}
strong {{ color: #132e3b; }}
table {{ border-collapse: collapse; width: 100%; table-layout: fixed;
  margin: 7pt 0 11pt; font-size: 7.15pt; line-height: 1.38; }}
thead {{ display: table-header-group; }}
tr {{ break-inside: avoid; }}
th, td {{ border-bottom: .55pt solid #cbd7dc; padding: 4pt 3pt;
  overflow-wrap: anywhere; vertical-align: top; }}
th {{ background: #eaf1f3; color: #183f50; font-weight: bold; }}
tbody tr:nth-child(even) {{ background: #f6f9fa; }}
td:first-child, th:first-child {{ width: 25%; text-align: left; }}
figure {{ margin: 10pt 0 13pt; break-inside: avoid; }}
figure img {{ display: block; width: 100%; height: auto; }}
figcaption {{ margin-top: 4pt; font-size: 8pt; line-height: 1.4; color: #415663; }}
a {{ color: #175e76; text-decoration: none; }}
code {{ font: 8pt "Nimbus Mono PS", "Response CJK", monospace;
  overflow-wrap: anywhere; }}
ul, ol {{ padding-left: 17pt; margin: 0 0 8pt; }}
</style></head><body>{body}</body></html>'''
HTML(string=html, base_url=str(ROOT)).write_pdf(target)
print(target)
