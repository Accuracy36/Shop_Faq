"""Helpers that reproduce the lab4 report's paragraph/table XML on top of its styles."""
import re
from pathlib import Path
from xml.sax.saxutils import escape

import docx
from docx.oxml import parse_xml
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.shared import Emu, Mm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH

NS = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
      'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math"')
TEXT_W = 9026            # A4 width minus 1440 dxa margins on both sides
EMU_PER_FIG_IN = 5334000 / 7.2   # lab4 scaled a 7.2 in figure to 5334000 EMU


def rpr(font='Times New Roman', size=22, bold=False, east='宋体'):
    b = '<w:b/>' if bold else '<w:b w:val="0"/>'
    return (f'<w:rPr><w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}" '
            f'w:eastAsia="{east}"/>{b}<w:color w:val="000000"/><w:sz w:val="{size}"/>'
            '<w:lang w:val="en-US" w:eastAsia="zh-CN"/></w:rPr>')


def t(text):
    if '\n' in text:
        return '<w:br/>'.join(t(part) for part in text.split('\n'))
    s = escape(text)
    if text != text.strip() or '  ' in text:
        return f'<w:t xml:space="preserve">{s}</w:t>'
    return f'<w:t>{s}</w:t>'


def run(text, **kw):
    return f'<w:r>{rpr(**kw)}{t(text)}</w:r>'


class Report:
    def __init__(self, template):
        self.doc = docx.Document(template) if Path(template).exists() else docx.Document()
        if 'CompleteSourceCode' not in self.doc.styles:
            from docx.enum.style import WD_STYLE_TYPE
            self.doc.styles.add_style('CompleteSourceCode', WD_STYLE_TYPE.PARAGRAPH)
        body = self.doc.element.body
        self.sect = body[-1]
        for el in list(body)[:-1]:
            body.remove(el)
        for rid, rel in list(self.doc.part.rels.items()):
            if rel.reltype == RT.IMAGE:
                self.doc.part.drop_rel(rid)
        pkg = self.doc.part.package
        for rid, rel in list(pkg.rels.items()):
            if rel.reltype == RT.THUMBNAIL:
                del pkg.rels[rid]
        # The user's traditional homework layout takes precedence over template defaults.
        for section in self.doc.sections:
            section.page_width, section.page_height = Mm(210), Mm(297)
            section.left_margin = section.right_margin = Pt(72)
            section.top_margin = section.bottom_margin = Pt(62)
            section.header_distance, section.footer_distance = Pt(25), Pt(28)
            section.different_first_page_header_footer = False
            for hf in (section.header, section.footer):
                for child in list(hf._element):
                    hf._element.remove(child)
            footer = section.footer.add_paragraph()
            footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
            footer.paragraph_format.first_line_indent = Pt(0)
            field = OxmlElement('w:fldSimple')
            field.set(qn('w:instr'), 'PAGE')
            footer._p.append(field)
            footer.style = self.doc.styles['Normal']
        for style in self.doc.styles:
            if hasattr(style, 'font'):
                style.font.color.rgb = RGBColor(0, 0, 0)
            for border in list(style.element.xpath('.//w:pBdr')):
                border.getparent().remove(border)
        for name, size, leading in [('Normal', 11, 17.5), ('Title', 17, 23),
                                    ('Heading 1', 12, 18), ('CompleteSourceCode', 9, 12)]:
            style = self.doc.styles[name]
            style.font.name = 'Consolas' if name == 'CompleteSourceCode' else 'Times New Roman'
            style.font.size = Pt(size)
            style.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'), '宋体')
            style.paragraph_format.line_spacing = Pt(leading)
            style.paragraph_format.space_after = Pt(5 if name == 'Normal' else 0)
            style.paragraph_format.page_break_before = False
            style.paragraph_format.first_line_indent = Pt(22 if name == 'Normal' else 0)
            style.font.bold = name == 'Heading 1'
        self.doc.styles['Heading 1'].paragraph_format.space_before = Pt(9)
        self.doc.styles['Heading 1'].paragraph_format.space_after = Pt(5)
        self.doc.styles['Heading 1'].paragraph_format.keep_with_next = True
        self.doc.styles['Title'].paragraph_format.space_after = Pt(10)

    def add(self, xml):
        el = parse_xml(xml.replace('<w:p>', f'<w:p {NS}>', 1)
                       .replace('<w:tbl>', f'<w:tbl {NS}>', 1))
        self.sect.addprevious(el)
        return el

    # --- paragraphs -----------------------------------------------------
    def title(self, text):
        self.add(f'<w:p><w:pPr><w:pStyle w:val="Title"/><w:jc w:val="center"/></w:pPr>'
                 f'<w:r>{t(text)}</w:r></w:p>')

    def idline(self, text):
        self.add('<w:p><w:pPr><w:spacing w:after="240"/><w:ind w:firstLine="0"/>'
                 '<w:jc w:val="center"/><w:kinsoku w:val="1"/><w:overflowPunct w:val="1"/>'
                 f'</w:pPr>{run(text)}</w:p>')

    def para(self, text, indent=True, keep=False):
        ind = '' if indent else '<w:ind w:firstLine="0"/>'
        keep = '<w:keepNext/>' if keep else ''
        self.add(f'<w:p><w:pPr>{ind}{keep}<w:jc w:val="left"/><w:kinsoku w:val="1"/>'
                 f'<w:overflowPunct w:val="1"/></w:pPr>{run(text)}</w:p>')

    def h1(self, text):
        self.add(f'<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r>{t(text)}</w:r></w:p>')

    def table_caption(self, text, before=0):
        self.add(f'<w:p><w:pPr><w:spacing w:line="300" w:lineRule="exact" w:before="{before}" w:after="140"/>'
                 '<w:ind w:firstLine="0"/><w:jc w:val="center"/><w:keepNext/><w:kinsoku w:val="1"/>'
                 f'<w:overflowPunct w:val="1"/></w:pPr>{run(text, size=20)}</w:p>')

    def figure_caption(self, text):
        self.add('<w:p><w:pPr><w:spacing w:line="300" w:lineRule="exact" w:after="140"/>'
                 '<w:ind w:firstLine="0"/><w:jc w:val="center"/><w:keepNext w:val="0"/>'
                 '<w:kinsoku w:val="1"/><w:overflowPunct w:val="1"/></w:pPr>'
                 f'{run(text, size=20)}</w:p>')

    def figure(self, path, fig_width_in, caption):
        p = self.doc.add_paragraph()
        p.add_run().add_picture(str(path), width=Emu(round(fig_width_in * EMU_PER_FIG_IN)))
        ppr = parse_xml(f'<w:pPr {NS}><w:keepNext/><w:spacing w:line="240" w:lineRule="auto" '
                        'w:before="60" w:after="40"/><w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>')
        p._p.insert(0, ppr)
        self.figure_caption(caption)

    def code(self, lines, size=18, line=250, east='宋体'):
        lines = list(lines)
        for i, s in enumerate(lines):
            last = i == len(lines) - 1
            keep = '<w:keepNext w:val="0"/>' if last else '<w:keepNext/>'
            after = 140 if last else 0
            self.add(f'<w:p><w:pPr><w:spacing w:line="{line}" w:lineRule="exact" w:after="{after}"/>'
                     f'<w:ind w:firstLine="0"/><w:jc w:val="left"/>{keep}<w:keepLines/>'
                     '<w:kinsoku w:val="1"/><w:overflowPunct w:val="1"/></w:pPr>'
                     f'{run(s, font="Consolas", size=size, east=east)}</w:p>')

    def math(self, omml):
        self.add('<w:p><w:pPr><w:spacing w:line="240" w:lineRule="auto" w:after="140"/>'
                 f'<w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr><m:oMath>{omml}</m:oMath></w:p>')

    def refs_heading(self, text):
        self.add('<w:p><w:pPr><w:ind w:firstLine="0"/><w:jc w:val="left"/><w:kinsoku w:val="1"/>'
                 f'<w:overflowPunct w:val="1"/></w:pPr>{run(text)}</w:p>')

    def ref(self, text):
        self.add('<w:p><w:pPr><w:spacing w:line="280" w:lineRule="exact"/><w:ind w:firstLine="0"/>'
                 '<w:jc w:val="left"/><w:kinsoku w:val="1"/><w:overflowPunct w:val="1"/></w:pPr>'
                 f'{run(text, size=18)}</w:p>')

    def appendix_label(self, text):
        self.add('<w:p><w:pPr><w:spacing w:before="160"/><w:ind w:firstLine="0"/><w:jc w:val="left"/>'
                 '<w:keepNext/><w:kinsoku w:val="1"/><w:overflowPunct w:val="1"/></w:pPr>'
                 f'{run(text, size=20, bold=True)}</w:p>')

    def source(self, lines, size=18, east='宋体', line=None):
        for s in lines:
            s = s.rstrip('\n').expandtabs(4)
            n = len(s) - len(s.lstrip(' '))
            ind = 240 + 99 * n
            spacing = f'<w:spacing w:line="{line}" w:lineRule="exact" w:before="0" w:after="0"/>' if line else ''
            self.add(f'<w:p><w:pPr><w:pStyle w:val="CompleteSourceCode"/><w:keepNext w:val="0"/>'
                     f'{spacing}<w:ind w:left="{ind}" w:hanging="{ind}"/><w:snapToGrid w:val="0"/>'
                     f'<w:wordWrap w:val="1"/></w:pPr>'
                     f'{run(s, font="Consolas", size=size, east=east)}</w:p>')

    # --- tables -----------------------------------------------------------
    def table(self, header, rows, widths, align=None):
        assert sum(widths) == TEXT_W, sum(widths)
        align = align or ['center'] * len(widths)
        grid = ''.join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        xml = ('<w:tbl><w:tblPr><w:tblW w:type="dxa" w:w="9026"/><w:jc w:val="center"/>'
               '<w:tblLayout w:type="fixed"/><w:tblLook w:firstColumn="1" w:firstRow="1" '
               'w:lastColumn="0" w:lastRow="0" w:noHBand="0" w:noVBand="1" w:val="04A0"/>'
               '<w:tblBorders><w:top w:val="single" w:sz="6" w:color="000000"/>'
               '<w:bottom w:val="single" w:sz="6" w:color="000000"/>'
               '<w:left w:val="nil" w:sz="6" w:color="000000"/><w:right w:val="nil" w:sz="6" w:color="000000"/>'
               '<w:insideH w:val="nil" w:sz="6" w:color="000000"/><w:insideV w:val="nil" w:sz="6" w:color="000000"/>'
               '</w:tblBorders><w:tblCellMar><w:top w:w="60" w:type="dxa"/><w:bottom w:w="60" w:type="dxa"/>'
               '<w:left w:w="40" w:type="dxa"/><w:right w:w="40" w:type="dxa"/></w:tblCellMar></w:tblPr>'
               f'<w:tblGrid>{grid}</w:tblGrid>')
        all_rows = [header] + rows
        for r, cells in enumerate(all_rows):
            head = r == 0
            trpr = '<w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>' if head else '<w:trPr><w:cantSplit/></w:trPr>'
            xml += f'<w:tr>{trpr}'
            for c, (text, w) in enumerate(zip(cells, widths)):
                border = ('<w:tcBorders><w:bottom w:val="single" w:sz="4" w:color="000000"/></w:tcBorders>'
                          if head else '')
                # As in lab4: only the header row keeps with the next row.
                keep = '<w:keepNext/>' if head else '<w:keepNext w:val="0"/>'
                xml += (f'<w:tc><w:tcPr><w:tcW w:type="dxa" w:w="{w}"/><w:vAlign w:val="center"/>{border}</w:tcPr>'
                        f'<w:p><w:pPr>{keep}<w:spacing w:line="300" w:lineRule="exact" w:after="0"/>'
                        f'<w:ind w:firstLine="0"/><w:jc w:val="{align[c]}"/></w:pPr>'
                        f'{run(str(text), size=20)}</w:p></w:tc>')
            xml += '</w:tr>'
        xml += '</w:tbl>'
        self.add(xml)

    def save(self, path):
        self.doc.save(str(path))


def omr(text):
    return ('<m:r><w:rPr><w:rFonts w:ascii="Cambria Math" w:hAnsi="Cambria Math" w:cs="Cambria Math" '
            'w:eastAsia="宋体"/><w:lang w:val="en-US" w:eastAsia="zh-CN"/><w:sz w:val="24"/></w:rPr>'
            f'<m:t xml:space="preserve">{escape(text)}</m:t></m:r>')


def omsub(base, sub):
    return f'<m:sSub><m:e>{omr(base)}</m:e><m:sub>{omr(sub)}</m:sub></m:sSub>'


def omfrac(num, den):
    return f'<m:f><m:num>{num}</m:num><m:den>{den}</m:den></m:f>'
