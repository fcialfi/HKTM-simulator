#!/usr/bin/env python3
"""Build the PDF user manual: docs/HKTM_Simulator_User_Manual.pdf.

    pip install reportlab
    python docs/manual/build_manual.py

The text lives in this file; screenshots are in docs/manual/img/. The CLI
option tables in Appendix A are read from the scripts' own argument parsers
at build time, so they always match the code. Retake the screenshots by
hand when the GUI layout changes.
"""

import argparse
import datetime
import os
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, CondPageBreak, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
IMG = os.path.join(HERE, "img")
OUT = os.path.join(ROOT, "docs", "HKTM_Simulator_User_Manual.pdf")
LOGO = os.path.join(ROOT, "assets", "exostaff_logo.png")

TITLE = "HKTM CCSDS Signal Generator"
SUBTITLE = "User Manual"
ISSUE = "Issue 1.0"
VERSION = "0.1.0"

ACCENT = colors.HexColor("#0a7ea4")
DARK = colors.HexColor("#14202b")
GREY = colors.HexColor("#5b6b78")
LIGHT = colors.HexColor("#eef4f7")
RULE = colors.HexColor("#c9d6de")


# --------------------------------------------------------------------------
# Fonts and styles
# --------------------------------------------------------------------------
def _register_fonts():
    """DejaVu if available (wide Unicode coverage), else the PDF base fonts."""
    cands = ["/usr/share/fonts/truetype/dejavu", "C:/Windows/Fonts", "/Library/Fonts"]
    names = {"": "DejaVuSans.ttf", "-Bold": "DejaVuSans-Bold.ttf", "-Mono": "DejaVuSansMono.ttf"}
    for d in cands:
        if all(os.path.exists(os.path.join(d, f)) for f in names.values()):
            for suffix, f in names.items():
                pdfmetrics.registerFont(TTFont("Body" + suffix, os.path.join(d, f)))
            pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold", italic="Body", boldItalic="Body-Bold")
            return "Body", "Body-Bold", "Body-Mono"
    return "Helvetica", "Helvetica-Bold", "Courier"


BODY, BOLD, MONO = _register_fonts()
ss = getSampleStyleSheet()
S = {
    "body": ParagraphStyle("body", parent=ss["Normal"], fontName=BODY, fontSize=9.5, leading=13.5,
                           spaceAfter=6, textColor=DARK),
    "small": ParagraphStyle("small", parent=ss["Normal"], fontName=BODY, fontSize=8, leading=10.5, textColor=DARK),
    "cell": ParagraphStyle("cell", parent=ss["Normal"], fontName=BODY, fontSize=8, leading=10.2, textColor=DARK),
    "cellb": ParagraphStyle("cellb", parent=ss["Normal"], fontName=BOLD, fontSize=8, leading=10.2,
                            textColor=colors.white),
    "cellmono": ParagraphStyle("cellmono", parent=ss["Normal"], fontName=MONO, fontSize=7.4, leading=9.6,
                               textColor=DARK),
    "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontName=BOLD, fontSize=17, leading=21, textColor=ACCENT,
                         spaceBefore=4, spaceAfter=10),
    "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontName=BOLD, fontSize=12.5, leading=16, textColor=DARK,
                         spaceBefore=10, spaceAfter=5),
    "h3": ParagraphStyle("h3", parent=ss["Heading3"], fontName=BOLD, fontSize=10.5, leading=14, textColor=DARK,
                         spaceBefore=7, spaceAfter=3),
    "bullet": ParagraphStyle("bullet", parent=ss["Normal"], fontName=BODY, fontSize=9.5, leading=13.5,
                             leftIndent=14, bulletIndent=4, spaceAfter=2.5, textColor=DARK),
    "caption": ParagraphStyle("caption", parent=ss["Normal"], fontName=BODY, fontSize=8, leading=10,
                              textColor=GREY, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10),
    "code": ParagraphStyle("code", parent=ss["Code"], fontName=MONO, fontSize=7.8, leading=10.2,
                           textColor=DARK, backColor=LIGHT, borderPadding=(5, 6, 5, 6), leftIndent=6,
                           rightIndent=6, spaceBefore=4, spaceAfter=9),
    "note": ParagraphStyle("note", parent=ss["Normal"], fontName=BODY, fontSize=9, leading=12.5,
                           textColor=DARK, backColor=colors.HexColor("#fff6e0"),
                           borderColor=colors.HexColor("#e8b54a"), borderWidth=0.6,
                           borderPadding=(5, 7, 5, 7), spaceBefore=4, spaceAfter=10),
    "toc1": ParagraphStyle("toc1", fontName=BOLD, fontSize=10, leading=14, leftIndent=0, textColor=DARK),
    "toc2": ParagraphStyle("toc2", fontName=BODY, fontSize=9, leading=11.5, leftIndent=16, textColor=DARK),
}


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
class _Counter:
    def __init__(self):
        self.h1 = 0
        self.h2 = 0
        self.fig = 0
        self.tab = 0


N = _Counter()


def H1(text, numbered=True):
    if numbered:
        N.h1 += 1
        N.h2 = 0
        text = f"{N.h1}  {text}"
    p = Paragraph(text, S["h1"])
    p.toc_level = 0
    return [CondPageBreak(9 * cm), Spacer(1, 6), p]


def H2(text):
    N.h2 += 1
    p = Paragraph(f"{N.h1}.{N.h2}  {text}", S["h2"])
    p.toc_level = 1
    return p


def H3(text):
    return Paragraph(text, S["h3"])


def P(text):
    return Paragraph(text, S["body"])


def NOTE(text):
    return Paragraph(text, S["note"])


def BUL(items):
    return [Paragraph(t, S["bullet"], bulletText="\u2022") for t in items]


def CODE(text):
    return Preformatted(text.strip("\n"), S["code"])


def FIG(name, caption, width_cm=16.0):
    path = os.path.join(IMG, name)
    img = Image(path)
    w = width_cm * cm
    img.drawHeight = img.imageHeight * w / img.imageWidth
    img.drawWidth = w
    N.fig += 1
    return KeepTogether([img, Paragraph(f"Figure {N.fig} \u2013 {caption}", S["caption"])])


def TABLE(rows, widths, caption=None, header=True, mono_cols=()):
    """rows: list of lists of str (markup allowed). First row is the header."""
    data = []
    for r, row in enumerate(rows):
        cells = []
        for c, txt in enumerate(row):
            if header and r == 0:
                style = S["cellb"]
            elif c in mono_cols:
                style = S["cellmono"]
            else:
                style = S["cell"]
            cells.append(Paragraph(txt, style))
        data.append(cells)
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("GRID", (0, 0), (-1, -1), 0.4, RULE),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), ACCENT)]
        st += [("BACKGROUND", (0, i), (-1, i), LIGHT) for i in range(2, len(data), 2)]
    t.setStyle(TableStyle(st))
    if caption:
        N.tab += 1
        cap = Paragraph(f"Table {N.tab} \u2013 {caption}", S["caption"])
        # keep short tables on one page with their caption
        return [KeepTogether([t, cap])] if len(rows) <= 12 else [t, cap]
    return [t, Spacer(1, 8)]


def esc(s):
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------
# CLI option tables, read from the scripts' own parsers
# --------------------------------------------------------------------------
class _Captured(Exception):
    def __init__(self, parser):
        self.parser = parser


def _capture_parser(module_name, entry):
    """Import a CLI script and grab its ArgumentParser when it calls parse_args()."""
    sys.path.insert(0, ROOT)
    orig = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        raise _Captured(self)

    argparse.ArgumentParser.parse_args = grab
    try:
        mod = __import__(module_name)
        getattr(mod, entry)()
    except _Captured as c:
        return c.parser
    finally:
        argparse.ArgumentParser.parse_args = orig
    raise RuntimeError(f"{module_name}.{entry}() did not parse arguments")


def option_rows(parser):
    rows = [["Option", "Default", "Description"]]
    for a in parser._actions:
        if isinstance(a, argparse._HelpAction):
            continue
        name = ", ".join(a.option_strings) if a.option_strings else a.dest
        if a.choices:
            name += " {" + ",".join(str(c) for c in a.choices) + "}"
        elif a.option_strings and a.nargs != 0:
            name += " " + (a.metavar or a.dest.upper())
        if isinstance(a, (argparse._StoreTrueAction, argparse._StoreFalseAction)):
            default = "off"
        elif not a.option_strings:
            default = "(required)"
        elif a.default is None:
            default = "\u2013"
        else:
            default = str(a.default)
        helptext = (a.help or "").replace("%%", "%")
        rows.append([esc(name), esc(default), esc(helptext) or "See section 5."])
    return rows


# --------------------------------------------------------------------------
# Page decoration
# --------------------------------------------------------------------------
def _cover(canv, doc):
    w, h = A4
    canv.saveState()
    canv.setFillColor(DARK)
    canv.rect(0, h - 11 * cm, w, 11 * cm, stroke=0, fill=1)
    canv.setFillColor(ACCENT)
    canv.rect(0, h - 11.25 * cm, w, 0.25 * cm, stroke=0, fill=1)
    if os.path.exists(LOGO):
        canv.drawImage(LOGO, 2 * cm, h - 4 * cm, width=6.2 * cm, height=6.2 * cm * 140 / 560,
                       mask="auto", preserveAspectRatio=True, anchor="sw")
    canv.setFillColor(colors.white)
    canv.setFont(BOLD, 28)
    canv.drawString(2 * cm, h - 7 * cm, TITLE)
    canv.setFont(BODY, 18)
    canv.setFillColor(colors.HexColor("#8fd3ea"))
    canv.drawString(2 * cm, h - 8.1 * cm, SUBTITLE)
    canv.setFont(BODY, 10)
    canv.setFillColor(colors.HexColor("#c8d6df"))
    canv.drawString(2 * cm, h - 9.4 * cm, "CCSDS 131.0-B-5 test signal generation and recorded-pass analysis")
    canv.drawString(2 * cm, h - 9.95 * cm, "for RF-Catcher (TestTree) Capture & Playback")
    # issue block
    canv.setFillColor(DARK)
    canv.setFont(BODY, 9.5)
    y = 6 * cm
    for k, v in (("Document", f"{TITLE} \u2013 {SUBTITLE}"), ("Software version", VERSION),
                 ("Issue", ISSUE), ("Date", datetime.date.today().strftime("%d %B %Y"))):
        canv.setFillColor(GREY)
        canv.drawString(2 * cm, y, k)
        canv.setFillColor(DARK)
        canv.drawString(6 * cm, y, v)
        y -= 0.6 * cm
    canv.restoreState()


def _page(canv, doc):
    w, h = A4
    canv.saveState()
    canv.setStrokeColor(RULE)
    canv.setLineWidth(0.5)
    canv.line(2 * cm, h - 1.5 * cm, w - 2 * cm, h - 1.5 * cm)
    canv.line(2 * cm, 1.5 * cm, w - 2 * cm, 1.5 * cm)
    canv.setFont(BODY, 7.5)
    canv.setFillColor(GREY)
    canv.drawString(2 * cm, h - 1.3 * cm, f"{TITLE} \u2013 {SUBTITLE}")
    canv.drawRightString(w - 2 * cm, h - 1.3 * cm, f"{ISSUE} \u00b7 v{VERSION}")
    canv.drawString(2 * cm, 1.05 * cm, "Exostaff")
    canv.drawRightString(w - 2 * cm, 1.05 * cm, f"Page {doc.page}")
    canv.restoreState()


class ManualDoc(BaseDocTemplate):
    def __init__(self, path):
        super().__init__(path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2.1 * cm,
                         bottomMargin=2 * cm, title=f"{TITLE} \u2013 {SUBTITLE}", author="Exostaff",
                         subject="User manual", creator="docs/manual/build_manual.py")
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f")
        self.addPageTemplates([PageTemplate("cover", [frame], onPage=_cover),
                               PageTemplate("body", [frame], onPage=_page)])

    def afterFlowable(self, f):
        level = getattr(f, "toc_level", None)
        if level is not None:
            text = f.getPlainText()
            key = f"h{id(f)}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=level > 0)
            self.notify("TOCEntry", (level, text, self.page, key))


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------
def content():
    st = [NextPageTemplate("body"), PageBreak()]
    toc = TableOfContents()
    toc.levelStyles = [S["toc1"], S["toc2"]]
    st += [Paragraph("Contents", S["h1"]), toc, PageBreak()]

    # ---------------------------------------------------------------- 1
    st += H1("Introduction")
    st += [H2("Purpose"),
           P("The <b>HKTM CCSDS Signal Generator</b> produces baseband IQ test signals that follow the CCSDS "
             "telemetry channel-coding chain (CCSDS 131.0-B-5) and writes them in a format that an IQ "
             "recorder/replayer can play back at RF \u2013 in particular the RF-Catcher (TestTree) Capture "
             "&amp; Playback application. The signal can then be injected into a ground-station receiver to "
             "test demodulation, decoding, frame synchronisation and Virtual Channel routing with fully known "
             "content."),
           P("It also analyses real recorded passes (RF-Catcher <font face='%s'>.rfcatcher</font> files or raw "
             "IQ): it measures symbol rate, carrier offset, Es/N0 and impairments, decodes the frames, and can "
             "copy the measured parameters back into the generator so that a recorded downlink can be "
             "reproduced." % MONO),
           H2("What the tool provides"),
           ]
    st += BUL([
        "<b>Signal generator</b>: Reed-Solomon, pseudo-randomizer, ASM, convolutional coding with "
        "puncturing, NRZ-L, QPSK/BPSK, RRC pulse shaping, all configurable.",
        "<b>Test features</b>: synthetic TM primary headers with selectable Spacecraft ID and Virtual Channels, "
        "deterministic Reed-Solomon error injection, transmitter impairments (LO offset, phase noise, IQ "
        "imbalance, PA compression) and channel Doppler.",
        "<b>Outputs</b>: raw interleaved IQ (float32 or int16) with a JSON metadata file, or a ready-to-play "
        "RF-Catcher <font face='%s'>.rfcatcher</font> recording." % MONO,
        "<b>Recording analysis</b>: characterisation and decoding of real recorded passes.",
        "<b>Two interfaces</b>: a graphical interface (GUI) in the web browser with live spectrum and "
        "constellation, and command-line tools (CLI) for scripted use. Both use the same signal chain code.",
    ])
    st += [H2("Reference documents")]
    st += TABLE([
        ["Ref.", "Document"],
        ["[RD-1]", "CCSDS 131.0-B-5, TM Synchronization and Channel Coding, Blue Book, September 2023"],
        ["[RD-2]", "CCSDS 132.0-B-3, TM Space Data Link Protocol, Blue Book (primary header only)"],
        ["[RD-3]", "ECSS-E-ST-50-01C, Space data links \u2013 Telemetry synchronization and channel coding"],
        ["[RD-4]", "TestTree RF-Catcher user manual (Capture &amp; Playback application)"],
        ["[RD-5]", "PACKAGING.md and README.md in the software repository"],
    ], [2.2, 14.8])
    st += [H2("Abbreviations")]
    st += TABLE([
        ["Term", "Meaning"],
        ["ASM", "Attached Sync Marker (0x1ACFFC1D)"],
        ["CADU", "Channel Access Data Unit: ASM + Reed-Solomon codeblock(s)"],
        ["Es/N0", "Energy per symbol to noise power spectral density ratio"],
        ["FER", "Frame Error Rate"],
        ["IQ", "In-phase / Quadrature baseband samples"],
        ["NRZ-L", "Non-Return-to-Zero Level line coding"],
        ["PA", "Power Amplifier"],
        ["PSD", "Power Spectral Density"],
        ["QPSK / BPSK", "Quadrature / Binary Phase Shift Keying"],
        ["RRC", "Root-Raised-Cosine pulse-shaping filter (roll-off \u03b1)"],
        ["RS", "Reed-Solomon code"],
        ["SCID / VCID", "Spacecraft Identifier (10 bits) / Virtual Channel Identifier (3 bits)"],
    ], [3, 14])

    # ---------------------------------------------------------------- 2
    st += H1("Installation and start-up")
    st += [P("The tool can be used in two ways: as a <b>standalone Windows executable</b> (no Python needed), or "
             "<b>from the source code</b> with Python. The functions are the same."),
           H2("Standalone executable"),
           H3("What you receive"),
           P("A zipped folder <font face='%s'>HKTM-CCSDS-Signal-Generator</font> and, separately, a licence "
             "file <font face='%s'>license.key</font>." % (MONO, MONO)),
           H3("Installation")]
    st += BUL([
        "Unzip the folder anywhere (for example <font face='%s'>C:\\Tools\\</font>). No installer and no "
        "administrator rights are needed." % MONO,
        "Copy <font face='%s'>license.key</font> into that folder, next to "
        "<font face='%s'>HKTM-CCSDS-Signal-Generator.exe</font>." % (MONO, MONO),
        "Keep the <font face='%s'>_internal</font> folder next to the <font face='%s'>.exe</font>: it contains "
        "Python and the libraries. The folder is about 500 MB." % (MONO, MONO),
    ])
    st += [CODE(r"""
HKTM-CCSDS-Signal-Generator\
    HKTM-CCSDS-Signal-Generator.exe
    license.key            <- the licence file you received
    _internal\             <- do not move or delete
"""),
           H3("Starting the program")]
    st += BUL([
        "Double-click <font face='%s'>HKTM-CCSDS-Signal-Generator.exe</font>." % MONO,
        "A splash screen with the logo shows the start-up progress, and a console window opens.",
        "After a few seconds the GUI opens in the default web browser at "
        "<font face='%s'>http://localhost:8501</font>. If the browser does not open, type that address by hand "
        "(the console window also prints it)." % MONO,
        "The program runs only on your computer: the browser is just its display. Nothing is sent over the "
        "network.",
        "<b>To stop the program, close the console window.</b> Closing the browser tab alone leaves it "
        "running.",
    ])
    st += [H3("Licence"),
           P("The executable checks the licence file at every start. It holds the customer name and the last "
             "valid day, and is signed by the issuer: any change to the file makes it invalid. The sidebar of "
             "the GUI shows <i>Licensed to &lt;customer&gt; \u00b7 valid until &lt;date&gt;</i>; during the last "
             "30 days a warning is shown instead.")]
    st += TABLE([
        ["Situation", "What happens", "What to do"],
        ["No <font face='%s'>license.key</font> in the folder" % MONO,
         "The console prints \u201cNo licence found\u201d with the folder to use, and the program stops.",
         "Copy the licence file next to the <font face='%s'>.exe</font>." % MONO],
        ["Licence expired", "The console prints the expiry date and the program stops.",
         "Ask the issuer for a new licence file (no new executable is needed)."],
        ["File altered or damaged", "\u201cThe licence file is not valid\u201d / \u201cdamaged\u201d.",
         "Use the original file as received."],
        ["System date set back", "\u201cThe system date appears to have been set back\u201d.",
         "Correct the date and time of the computer and restart."],
        ["Expires within 30 days", "The program works; the GUI shows a warning.", "Request a renewal."],
    ], [4.2, 6.6, 6.2], caption="Licence messages at start-up")
    st += [H2("Running from source"),
           P("Requires Python 3.10 or later. In a terminal, in the repository folder:"),
           CODE("""
pip install -r requirements.txt
streamlit run app.py            # GUI, opens in the browser
python generate_signal.py -h    # CLI help
"""),
           P("When run from source there is no licence check.")]

    # ---------------------------------------------------------------- 3
    st += H1("The signal chain")
    st += [P("The generator applies the CCSDS channel-coding steps in the order of [RD-1]. Each step can be "
             "configured or switched off:"),
           CODE("""
payload -> RS(255,223) interleave x5 -> [pseudo-randomizer, ASM excluded]
        -> + ASM (per CADU)  -> convolutional K=7 r=1/2 (over the CADU stream)
        -> NRZ-L -> QPSK (Gray) or BPSK -> RRC or no filter -> [impairments] -> [Doppler]
        -> normalisation -> [resampling] -> IQ file
"""),
           H2("Baseline parameters")]
    st += TABLE([
        ["Parameter", "Baseline value", "Options"],
        ["Modulation", "QPSK (Gray)", "QPSK, BPSK"],
        ["Symbol rate", "1 785 000 S/s", "any; or set the bit rate (bit rate = symbol rate \u00d7 2 for QPSK)"],
        ["Bit rate (channel)", "3 570 000 bps", "derived"],
        ["Line coding", "NRZ-L (1 \u2192 +1, 0 \u2192 \u22121)", "NRZ-L only"],
        ["Reed-Solomon", "RS(255,223), E = 16, interleave depth I = 5", "on/off; E = 8 or 16; I = 1, 2, 3, 4, 5, 8"],
        ["Pseudo-randomizer", "CLI: none \u00b7 GUI: long",
         "none; short (255-bit, legacy); long (131071-bit, CCSDS default since 2023)"],
        ["ASM", "0x1ACFFC1D, 4 bytes, not coded by RS", "fixed"],
        ["CADU length", "4 + 255 \u00d7 I = 1279 bytes", "depends on I"],
        ["Convolutional", "K = 7, rate 1/2, G1 = 171o, G2 = 133o, G2 inverted",
         "on/off; rate 1/2, 2/3, 3/4, 5/6, 7/8"],
        ["Pulse shaping", "RRC, \u03b1 = 0.35, span 8 symbols, 4 samples/symbol",
         "RRC (\u03b1, span) or none (rectangular NRZ pulses); samples/symbol"],
        ["Spacecraft ID", "0x123 (291)", "0\u20131023; written only with Virtual Channels enabled"],
        ["Output", "float32 IQ, peak 0.9, native rate (7.14 MS/s)",
         "int16, .rfcatcher, peak amplitude, resampling to a fixed rate"],
    ], [3.4, 6.2, 7.4], caption="Baseline parameters")
    st += [H2("Input formats"),
           P("The payload can be pseudo-random (reproducible with a seed) or read from a file. The input format "
             "says what the data already contains, so that no step is applied twice:")]
    st += TABLE([
        ["Input format", "File content", "Steps applied by the generator"],
        ["Transfer Frame (default)", "Raw, uncoded Transfer Frames", "RS, randomizer, ASM, convolutional, modulation"],
        ["ASM + Transfer Frame", "ASM + frame, not scrambled (e.g. decoded output of a ground station, or frames "
         "saved by the recording analysis)",
         "Randomizer (never on the ASM; default short, \u201cnone\u201d not allowed), convolutional, modulation. "
         "No RS, no new ASM."],
        ["CADU as on air", "ASM + already RS-coded and scrambled block", "Convolutional and modulation only "
         "(randomizer must be \u201cnone\u201d)"],
    ], [3.8, 6.4, 6.8], caption="Input formats")
    st += [P("With file input, each record is exactly 4 + 255 \u00d7 I bytes. The tool searches for the first real "
             "ASM, so a file need not start on a CADU boundary, and skips any recorder-added bytes between the end "
             "of one CADU and the next ASM.")]

    # ---------------------------------------------------------------- 4
    st += H1("Graphical interface (GUI)")
    st += [P("The GUI has a sidebar with the generator settings and two tabs: <b>Signal generator</b> and "
             "<b>Recording analysis</b>. Every change of a setting recomputes a short preview of the signal "
             "immediately."),
           FIG("gui_overview.png", "Main window: settings sidebar, header with the active baseline, the two tabs "
               "and the live preview of the signal generator"),
           H2("Settings sidebar"),
           P("The settings are grouped in the order the chain applies them. Each field has a help icon (?) with "
             "a short explanation.")]
    side = TABLE([
        ["Section", "Content"],
        ["1 \u00b7 Modulation &amp; symbol rate", "QPSK/BPSK, line coding, symbol rate or bit rate (the other is "
         "derived)."],
        ["2 \u00b7 Input format", "What the payload contains (Table 3)."],
        ["3 \u00b7 Channel coding", "Reed-Solomon on/off, E, interleave depth; convolutional on/off, code rate, G2 "
         "inversion; pseudo-randomizer."],
        ["4 \u00b7 Payload &amp; duration", "Pseudo-random (seed) or file; preview CADU count; <b>Assign Virtual "
         "Channels</b> (VCID list) and <b>Spacecraft ID</b>."],
        ["5 \u00b7 Deterministic error injection", "Exact number of RS symbol errors in a chosen codeword of chosen "
         "CADUs or of one Virtual Channel."],
        ["6 \u00b7 Pulse shaping", "RRC filter (roll-off \u03b1, span) or <b>None (NRZ, rectangular)</b>; "
         "samples per symbol."],
        ["7 \u00b7 Impairments &amp; Doppler", "LO frequency offset and phase noise; IQ gain and phase imbalance; PA "
         "saturation and AM-PM; channel Doppler shift and rate."],
    ], [4.6, 7.4])
    side_tbl = Table([[side[0], Image(os.path.join(IMG, "gui_sidebar.png"), width=4.0 * cm,
                                      height=4.0 * cm * 1060 / 450)]],
                     colWidths=[12.3 * cm, 4.7 * cm])
    side_tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    N.fig += 1
    st += [side_tbl, Paragraph(f"Figure {N.fig} \u2013 Sidebar sections (right: section 6 with the unfiltered NRZ "
                               "pulse shape selected)", S["caption"])]
    st += [H2("Signal generator tab: live preview"),
           P("The <b>Live preview</b> panel shows:")]
    st += BUL([
        "the chain stages as a row of labels: active stages are highlighted, disabled ones are greyed out;",
        "the theoretical occupied bandwidth (RRC: Rs \u00d7 (1 + \u03b1); no filter: 2 \u00d7 Rs), duration, sample rate, number of symbols and "
        "compute time of the preview;",
        "the <b>spectrum</b> (PSD) with the theoretical band edges as dashed red lines;",
        "the <b>constellation</b>, taken after a matched filter, so impairments (phase noise, IQ imbalance, PA) "
        "are visible;",
        "an <b>I/Q excerpt</b> over time.",
    ])
    st += [FIG("gui_plots.png", "Live preview: spectrum with theoretical band edges, constellation and I/Q "
               "excerpt"),
           NOTE("The preview uses only a few tens of CADUs to stay fast. The length of the exported file is set "
                "separately in the Export panel."),
           H2("Signal generator tab: export"),
           P("The <b>Export</b> panel writes the full-length file:")]
    st += TABLE([
        ["Field", "Meaning"],
        ["Export CADU count", "Length of the file. The panel shows the resulting duration and file size."],
        ["Output format", "<b>RF-Catcher recording (.rfcatcher)</b>: carries its own sample rate, RF frequency and "
         "bandwidth. <b>Raw IQ int16</b> (RF-Catcher sample format) or <b>raw IQ float32</b>: no header; rate and "
         "frequency must be set by hand on the replayer."],
        ["Peak amplitude", "Normalised peak (default 0.9), headroom against clipping."],
        ["Resample to fixed rate", "Resample to an exact rate accepted by the replayer, e.g. 10 000 000 Hz. "
         "Without it, the native rate is symbol rate \u00d7 4."],
        ["RF center frequency", "Written into the .rfcatcher metadata (default 1707 MHz)."],
        ["Bandwidth: automatic", "RF-Catcher's linked value, sample rate / 1.1; untick to enter it."],
        ["Device metadata from", "Optional real .rfcatcher recording whose device fields (serial number, "
         "firmware, gain) are copied."],
        ["Generate export file", "Creates the file in the <font face='%s'>output</font> folder and offers the "
         "download buttons (file and metadata). Files above 1 GB stay in the folder only." % MONO],
    ], [4.2, 12.8], caption="Export panel")
    st += [FIG("gui_export.png", "Export panel with the .rfcatcher output format"),
           NOTE("RF-Catcher only accepts some sample rates (divisions of its ADC clock). If the replayer refuses "
                "the native rate, tick <b>Resample to fixed rate</b> and use 10 000 000 Hz. The panel warns when "
                "the rate differs from the template recording."),
           H2("Recording analysis tab"),
           P("This tab characterises a real recorded pass and can transfer the result to the generator.")]
    st += BUL([
        "Enter the <b>local path</b> of the recording (.rfcatcher or raw IQ). The file is read directly from "
        "disk, not uploaded, so multi-GB files are fine.",
        "Choose the slice: <b>Start at</b> and <b>Duration</b> (up to 2 s; 1 s is plenty). Only that slice is "
        "read.",
        "For raw IQ, give the sample rate and sample format; for .rfcatcher they come from the metadata.",
        "Nominal symbol rate and matched-filter roll-off are starting values for the estimation.",
        "<b>Decode frames</b> runs the Viterbi decoder and the frame analysis (slower: about 15 s).",
        "Click <b>Analyze recording</b>.",
    ])
    st += [FIG("gui_analysis_input.png", "Recording analysis: input fields", width_cm=13.5),
           P("The results show symbol rate (and offset in ppm), carrier offset and drift, Es/N0 and EVM, the "
             "number of discrete spectral lines, the randomizer and CADU length found, the share of idle frames "
             "and the Spacecraft ID; then the spectrum, the constellation after carrier recovery and the residual "
             "carrier phase. <b>Full report</b> lists all values, including IQ imbalance, phase noise and the "
             "Virtual Channels found."),
           FIG("gui_analysis_results.png", "Recording analysis: results and the \u201cReproduce it in the "
               "generator\u201d panel", width_cm=13.5),
           H3("Reproduce it in the generator"),
           P("Tick the values to copy (symbol rate, rounded to 50 Hz; modulation; randomizer; optionally the "
             "carrier offset) and click <b>Apply to generator</b>: the generator tab opens with those settings. "
             "<b>Download the decoded frames</b> saves them as ASM + Transfer Frame records; load that file as "
             "payload with input format \u201cASM + Transfer Frame\u201d to regenerate the recorded content bit "
             "for bit.")]

    # ---------------------------------------------------------------- 5
    st += H1("Command-line tools (CLI)")
    st += [P("Three scripts cover the same functions as the GUI. All options are listed in Appendix A and by "
             "<font face='%s'>-h</font>." % MONO),
           H2("generate_signal.py \u2013 generate a signal"),
           CODE("""
python generate_signal.py                       # baseline, file in output/
python generate_signal.py --n-cadu 500 --randomizer long -o test.raw
python generate_signal.py --rs-e 8 --interleave-depth 2 --conv-rate 3/4 -o test2.raw
python generate_signal.py --modulation BPSK -o test3.raw
python generate_signal.py --dtype int16 --target-fs 10e6 --peak 0.9
python generate_signal.py --rfcatcher --target-fs 10e6 --rf-frequency-mhz 1707
"""),
           P("Without <font face='%s'>-o</font>, the file is written to "
             "<font face='%s'>output/qpsk_ccsds_&lt;fs&gt;Msps_&lt;n_cadu&gt;cadu_&lt;timestamp&gt;.iq</font>. "
             "Generation runs in batches, so memory use does not grow with the file length (resampling is the "
             "only exception: very long files with resampling are refused before starting)." % (MONO, MONO)),
           H2("analyze_recording.py \u2013 analyse a recording"),
           CODE(r"""
python analyze_recording.py C:\RF-Catcher\AWS_2_split.rfcatcher --offset 30 --duration 1 --plot aws2.png
python analyze_recording.py capture.iq --fs 10e6 --dtype int16 --save-frames frames.bin
"""),
           P("It prints the same report as the GUI. <font face='%s'>--plot</font> saves spectrum, constellation "
             "and residual phase as PNG; <font face='%s'>--save-frames</font> writes the decoded frames in the "
             "\u201cASM + Transfer Frame\u201d input format. About 15 s and 2 GB of memory per second of signal, "
             "plus about 15 s of decoding (limited by default to the first 400 000 symbols, about 40 CADUs)."
             % (MONO, MONO)),
           H2("verify_spectrum.py \u2013 check an exported file"),
           CODE("python verify_spectrum.py output/qpsk_ccsds_....iq --plot spectrum.png"),
           P("Computes the PSD of a generated file (reading its .meta.json for rate and format) and measures the "
             "\u22123 dB and null-to-null bandwidths. For the RRC baseline (Rs = 1.785 MS/s, \u03b1 = 0.35): "
             "\u22123 dB \u2248 Rs, null-to-null \u2248 Rs (1 + \u03b1) \u2248 2.41 MHz. Without filter (NRZ): "
             "\u22123 dB \u2248 0.886 \u00d7 Rs, null-to-null = 2 \u00d7 Rs \u2248 3.57 MHz.")]

    # ---------------------------------------------------------------- 6
    st += H1("Test scenarios")
    st += [P("Typical uses, with the CLI command. The same settings are available in the GUI sidebar."),
           H2("Virtual Channels and Spacecraft ID"),
           P("Builds a real 6-byte TM primary header [RD-2] into each frame: Spacecraft ID, Virtual Channel ID "
             "(round-robin over the list), master and virtual channel frame counters. Repeating a VCID gives it "
             "a larger share. Use it to check that a receiver identifies and routes frames per Virtual Channel."),
           CODE("python generate_signal.py --vcid-list 0,0,1,2 --spacecraft-id 291 -o vc_test.raw"),
           NOTE("The Spacecraft ID (10 bits, 0\u20131023, default 0x123 = 291) is written only together with "
                "Virtual Channels (<font face='%s'>--vcid-list</font> / \u201cAssign Virtual Channels\u201d). It "
                "cannot be combined with a payload file, because it would overwrite the first 6 bytes of real "
                "data." % MONO),
           H2("Reed-Solomon boundary test"),
           P("Flips an exact number of RS symbols in one codeword of chosen CADUs. With E = 16, 16 errors must be "
             "corrected; 17 must be detected as uncorrectable and never miscorrected."),
           CODE("""
python generate_signal.py --corrupt-rs-symbols 16 --corrupt-cadu-indices 10 -o rs16.raw
python generate_signal.py --corrupt-rs-symbols 17 --corrupt-cadu-indices 10 -o rs17.raw
"""),
           H2("Per-Virtual-Channel error accounting"),
           CODE("python generate_signal.py --vcid-list 0,1,2 --corrupt-rs-symbols 17 --corrupt-vc 1 -o vc1_err.raw"),
           P("Every VC 1 frame is uncorrectable; the receiver's FER report must show errors on VC 1 only."),
           H2("Transmitter impairments"),
           CODE("""
python generate_signal.py --freq-offset-hz 5000 --iq-gain-imbalance-db 0.8 \\
    --iq-phase-imbalance-deg 3 --phase-noise-linewidth-hz 200 -o tx_impair.raw
python generate_signal.py --pa-backoff-db -6 --pa-smoothness 3 --pa-am-pm-deg-per-db 3 -o pa_sat.raw
"""),
           P("Checks carrier recovery, image rejection and tolerance to EVM. A PA driven into saturation produces "
             "spectral regrowth outside the occupied band."),
           H2("Unfiltered NRZ signal"),
           P("Some satellites (AWS, for example) transmit QPSK without pulse shaping: rectangular NRZ pulses, "
             "followed at most by an output band-pass filter. Select <b>None (NRZ, rectangular)</b> in sidebar "
             "section 6, or on the CLI:"),
           CODE("python generate_signal.py --pulse-shape nrz --sps 8 -o nrz.raw"),
           P("The spectrum is a sinc\u00b2: first nulls at \u00b1Rs (null-to-null 2 \u00d7 Rs = 3.57 MHz, the same main "
             "lobe as an RRC with \u03b1 = 1) and sidelobes at about \u221213, \u221218, \u221221 dB that decay slowly and are "
             "not band-limited. Use 8 or more samples per symbol: at 4, the sidelobes above half the sample rate fold "
             "back into the band. Resampling to a lower rate removes them above the new Nyquist frequency. The "
             "constellation is computed with an integrate-and-dump (rectangular) matched filter."),
           FIG("gui_nrz.png", "Live preview with no filter: sinc\u00b2 spectrum with the first nulls marked, "
               "rectangular I/Q waveform", width_cm=14.5),
           H2("Doppler of a pass"),
           CODE("python generate_signal.py --doppler-hz 30000 --doppler-rate-hz-s -400 -o doppler.raw"),
           P("Frequency shift at the start of the file plus a constant rate (Hz/s): a linear approximation of the "
             "Doppler curve of a LEO pass, to check the receiver's carrier tracking."),
           H2("Reproducing a recorded pass"),
           ]
    st += BUL([
        "Analyse 1 s of the recording (GUI Recording analysis tab, or "
        "<font face='%s'>analyze_recording.py --save-frames frames.bin</font>)." % MONO,
        "Apply symbol rate, modulation and randomizer to the generator.",
        "Load the decoded frames as payload with input format \u201cASM + Transfer Frame\u201d.",
        "Export as .rfcatcher at the recorder's sample rate and replay it.",
    ])

    # ---------------------------------------------------------------- 7
    st += H1("Output files")
    st += TABLE([
        ["File", "Content"],
        ["<font face='%s'>*.iq</font> / <font face='%s'>*.raw</font> (float32)" % (MONO, MONO),
         "Interleaved I0, Q0, I1, Q1, \u2026 as 32-bit floats in [\u22121, +1]. No header."],
        ["<font face='%s'>*.iq</font> / <font face='%s'>*.raw</font> (int16)" % (MONO, MONO),
         "Interleaved 16-bit little-endian integers, 12 significant bits, range [\u22122048, 2047] (RF-Catcher "
         "format). No header."],
        ["<font face='%s'>*.meta.json</font>" % MONO,
         "All parameters used, actual sample rate and format of the file. Note: a raw file is baseband; the RF "
         "frequency must be set on the replayer."],
        ["<font face='%s'>*.rfcatcher</font>" % MONO,
         "RF-Catcher recording: a tar archive of <font face='%s'>&lt;name&gt;.iq</font> (int16) and "
         "<font face='%s'>&lt;name&gt;.json</font> (sample rate, RF frequency, bandwidth, duration, device "
         "fields). Headers are written exactly as RF-Catcher writes them." % (MONO, MONO)],
    ], [4.6, 12.4], caption="Output files")
    st += [P("All files are written to the <font face='%s'>output</font> folder unless another path is given." % MONO)]

    # ---------------------------------------------------------------- 8
    st += H1("Licence administration (issuer)")
    st += [P("This chapter is for the person who distributes the executable. Users only need section 2.1. "
             "The tool <font face='%s'>license_tool.py</font> is run from the source code and is not part of "
             "the executable." % MONO)]
    st += TABLE([
        ["Step", "Command", "Notes"],
        ["1. Once: key pair",
         "python license_tool.py keygen --private-key C:\\secure\\hktm_license_private_key.pem",
         "The private key stays with the issuer, outside the repository. The public key is written to "
         "assets/license_public_key.pem and is built into every executable made from then on."],
        ["2. Build", "pyinstaller hktm_simulator.spec", "See PACKAGING.md. Builds without the public key have no "
         "licence check."],
        ["3. Per customer",
         "python license_tool.py issue --private-key C:\\secure\\hktm_license_private_key.pem --customer "
         "\"Name\" --expires 2027-12-31 -o license.key",
         "The expiry date is the last valid day. Send license.key with the zipped folder."],
        ["4. Renewal", "Same as step 3 with a new date", "No new executable needed."],
        ["Check", "python license_tool.py verify license.key", "Prints VALID or the reason."],
    ], [2.6, 7.6, 6.8], caption="Licence administration", mono_cols=(1,))
    st += [NOTE("The licence is a deterrent, not copy protection. Never commit the private key or issued "
                "licence files; both are excluded by .gitignore. If the private key is lost, a new key pair, new "
                "licences and a rebuild are needed.")]

    # ---------------------------------------------------------------- 9
    st += H1("Limitations and troubleshooting")
    st += [H2("Known limitations")]
    st += BUL([
        "Turbo and LDPC codes (CCSDS 131.0-B-5 sections 6\u20138) are not implemented.",
        "Transfer Frame length constraints and RS virtual fill are not implemented: with QPSK, some combinations "
        "of punctured code rate and CADU count give an odd number of coded bits and are refused with an error. "
        "The baseline (rate 1/2) is never affected.",
        "Only NRZ-L with polarity 1 \u2192 +1; check the polarity expected by the receiver.",
        "Only the TM primary header is synthesised: no secondary header, no Space Packets, no frame CRC.",
        "Unfiltered NRZ pulses are ideal rectangles: the output filter of a real unshaped transmitter, which "
        "lowers the sidelobes (AWS: about \u221221 dB instead of \u221213 dB), is not modelled.",
        "The .rfcatcher output matches a real recording byte for byte, but playback on the instrument has not "
        "been confirmed yet.",
        "The RRC filter adds a transient of half the filter span at the start and end of the file.",
    ])
    st += [H2("Troubleshooting")]
    st += TABLE([
        ["Symptom", "Cause / action"],
        ["The executable starts slowly the first time", "Windows scans the new files; later starts take a few "
         "seconds. Wait for the splash screen to close."],
        ["Browser does not open", "Open http://localhost:8501 by hand. If the port is in use, close other "
         "instances of the program."],
        ["Program stops at start-up with a licence message", "See Table 1 (section 2.1)."],
        ["\u201codd number of coded bits\u201d error", "Change the CADU count or use convolutional rate 1/2 "
         "(see limitations)."],
        ["Error about CADU length with a payload file", "RS E and interleave depth do not match how the file's "
         "CADUs were built; set them to the real values."],
        ["Replayer refuses the sample rate", "Export with Resample to fixed rate, e.g. 10 000 000 Hz."],
        ["Recording analysis finds no ASM", "Wrong modulation or symbol rate, too low Es/N0, or a slice without "
         "signal; try another slice or switch off decoding to see the measurements."],
        ["An error in the console of the executable", "Report the full text of the console window to the "
         "supplier."],
    ], [5.6, 11.4], caption="Troubleshooting")

    # ---------------------------------------------------------------- A
    N.h1 = 0
    p = Paragraph("Appendix A  CLI option reference", S["h1"])
    p.toc_level = 0
    st += [PageBreak(), p,
           P("Generated from the scripts' own argument parsers when this manual was built.")]
    for title, mod, entry in (("generate_signal.py", "generate_signal", "build_cli"),
                              ("analyze_recording.py", "analyze_recording", "main"),
                              ("verify_spectrum.py", "verify_spectrum", "main")):
        h = Paragraph(f"A.{['generate_signal.py', 'analyze_recording.py', 'verify_spectrum.py'].index(title) + 1}"
                      f"  {title}", S["h2"])
        h.toc_level = 1
        st += [h]
        st += TABLE(option_rows(_capture_parser(mod, entry)), [5.4, 2.0, 9.6], mono_cols=(0, 1))
    return st


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    doc = ManualDoc(OUT)
    doc.multiBuild(content())
    print(f"Written {OUT} ({doc.page} pages)")


if __name__ == "__main__":
    main()
