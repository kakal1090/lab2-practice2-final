"""Render the generated Vietnamese Markdown report as a printable PDF."""
from pathlib import Path
import html,re,textwrap
from xml.sax.saxutils import escape
import markdown
from bs4 import BeautifulSoup
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,Preformatted,KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from .experiment import ROOT


def main():
    from matplotlib import get_data_path
    fonts=Path(get_data_path())/'fonts/ttf'
    pdfmetrics.registerFont(TTFont('DV',str(fonts/'DejaVuSans.ttf')))
    pdfmetrics.registerFont(TTFont('DV-Bold',str(fonts/'DejaVuSans-Bold.ttf')))
    pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-Bold',italic='DV',boldItalic='DV-Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='VN',fontName='DV',fontSize=9,leading=14,spaceAfter=8))
    styles.add(ParagraphStyle(name='VNH1',fontName='DV-Bold',fontSize=20,leading=26,textColor=colors.HexColor('#183b6b'),spaceAfter=18))
    styles.add(ParagraphStyle(name='VNH2',fontName='DV-Bold',fontSize=13,leading=18,textColor=colors.HexColor('#183b6b'),spaceBefore=14,spaceAfter=9,keepWithNext=True))
    styles.add(ParagraphStyle(name='VNH3',fontName='DV-Bold',fontSize=10,leading=15,spaceBefore=9,spaceAfter=6,keepWithNext=True))
    styles.add(ParagraphStyle(name='CellVN',fontName='DV',fontSize=7.1,leading=10))
    styles.add(ParagraphStyle(name='CodeVN',fontName='DV',fontSize=6.8,leading=9))
    source=(ROOT/'report/BAO_CAO_PRACTICE_2.md').read_text()
    soup=BeautifulSoup(markdown.markdown(source,extensions=['tables','fenced_code']),'html.parser')
    story=[];width=A4[0]-76
    for node in soup.children:
        tag=getattr(node,'name',None)
        if tag in ['h1','h2','h3']:
            story.append(Paragraph(escape(node.get_text()),styles[{'h1':'VNH1','h2':'VNH2','h3':'VNH3'}[tag]]))
        elif tag=='p':
            for img in node.find_all('img'):
                path=ROOT/'report'/img['src']
                from PIL import Image as PILImage
                with PILImage.open(path) as im:w,h=im.size
                draw_w=min(width,w);draw_h=draw_w*h/w
                story.extend([Image(str(path),width=draw_w,height=draw_h),Spacer(1,8)])
            txt=node.get_text(' ',strip=True)
            if txt:
                paragraph=Paragraph(escape(txt),styles['VN'])
                if txt=='Môi trường thực tế:':paragraph.keepWithNext=True
                story.append(paragraph)
        elif tag in ['ul','ol']:
            for li in node.find_all('li',recursive=False):
                story.append(Paragraph('• '+escape(li.get_text(' ',strip=True)),styles['VN']))
        elif tag=='table':
            rows=[]
            for tr in node.find_all('tr'):
                rows.append([Paragraph(escape(td.get_text(' ',strip=True)),styles['CellVN']) for td in tr.find_all(['td','th'])])
            n=len(rows[0]);colwidths=([width*.42]+[width*.58/(n-1)]*(n-1)) if n>1 else [width]
            table=Table(rows,colWidths=colwidths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9eff7')),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#d6dfe9')),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7f9fc')])]))
            story.extend([table,Spacer(1,10)])
        elif tag=='pre':
            text='\n'.join('\n'.join(textwrap.wrap(line,110,replace_whitespace=False) or ['']) for line in node.get_text().splitlines())
            story.append(Preformatted(text,styles['CodeVN']));story.append(Spacer(1,8))
    def page(c,doc):
        c.saveState();c.setFont('DV',7);c.setFillColor(colors.HexColor('#61728b'))
        c.drawString(38,22,'Practice 2 · CIFAR-10 · Thực nghiệm có log')
        c.drawRightString(A4[0]-38,22,f'Trang {doc.page}');c.restoreState()
    output=ROOT/'report/BAO_CAO_PRACTICE_2.pdf'
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=38,leftMargin=38,topMargin=36,bottomMargin=38,
                          title='Practice 2 - Báo cáo thực nghiệm pretrained networks',author='Nhóm Lab 2')
    doc.build(story,onFirstPage=page,onLaterPages=page)
    print(output)

if __name__=='__main__':main()
