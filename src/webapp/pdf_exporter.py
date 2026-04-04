import io
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT


# ─────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────

BLACK      = colors.HexColor('#0a0e1a')
DARK       = colors.HexColor('#0d1117')
BORDER     = colors.HexColor('#21262d')
TEXT_MAIN  = colors.HexColor('#1a1a2e')
TEXT_MUTED = colors.HexColor('#57606a')
BLUE       = colors.HexColor('#0969da')
RED        = colors.HexColor('#cf222e')
YELLOW     = colors.HexColor('#9a6700')
GREEN      = colors.HexColor('#1a7f37')
RED_BG     = colors.HexColor('#FFEBE9')
YELLOW_BG  = colors.HexColor('#FFF8C5')
GREEN_BG   = colors.HexColor('#DAFBE1')
WHITE      = colors.white


# ─────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────

def build_styles():
    custom = {
        'title': ParagraphStyle(
            'title',
            fontName='Helvetica-Bold',
            fontSize=22,
            textColor=TEXT_MAIN,
            spaceAfter=4,
            leading=26,
        ),
        'subtitle': ParagraphStyle(
            'subtitle',
            fontName='Helvetica',
            fontSize=10,
            textColor=TEXT_MUTED,
            spaceAfter=2,
        ),
        'section': ParagraphStyle(
            'section',
            fontName='Helvetica-Bold',
            fontSize=11,
            textColor=TEXT_MAIN,
            spaceBefore=16,
            spaceAfter=8,
        ),
        'body': ParagraphStyle(
            'body',
            fontName='Helvetica',
            fontSize=9,
            textColor=TEXT_MAIN,
            leading=14,
            spaceAfter=4,
        ),
        'mono': ParagraphStyle(
            'mono',
            fontName='Courier',
            fontSize=8,
            textColor=TEXT_MAIN,
            leading=12,
        ),
        'label': ParagraphStyle(
            'label',
            fontName='Helvetica-Bold',
            fontSize=8,
            textColor=TEXT_MUTED,
            spaceAfter=2,
        ),
        'small': ParagraphStyle(
            'small',
            fontName='Helvetica',
            fontSize=8,
            textColor=TEXT_MUTED,
            leading=11,
        ),
    }
    return custom


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def severity_color(severity):
    return {'HIGH': RED, 'MEDIUM': YELLOW, 'LOW': GREEN}.get(severity, TEXT_MUTED)

def severity_bg(severity):
    return {'HIGH': RED_BG, 'MEDIUM': YELLOW_BG, 'LOW': GREEN_BG}.get(severity, WHITE)

def section_rule():
    return HRFlowable(
        width='100%', thickness=1,
        color=BORDER, spaceAfter=8, spaceBefore=4
    )


# ─────────────────────────────────────────────
# SUMMARY CARDS
# ─────────────────────────────────────────────

def summary_card_table(summary, styles):
    cards = [
        ('TOTAL FLOWS', str(summary['total']),  BLUE),
        ('HIGH',        str(summary['high']),   RED),
        ('MEDIUM',      str(summary['medium']), YELLOW),
        ('LOW',         str(summary['low']),    GREEN),
    ]

    label_row = []
    value_row = []

    for label, value, color in cards:
        label_row.append(Paragraph(label, ParagraphStyle(
            'cl', fontName='Helvetica-Bold', fontSize=7,
            textColor=TEXT_MUTED, alignment=TA_CENTER
        )))
        value_row.append(Paragraph(value, ParagraphStyle(
            'cv', fontName='Helvetica-Bold', fontSize=28,
            textColor=color, alignment=TA_CENTER
        )))

    table = Table(
        [label_row, value_row],
        colWidths=[4.2 * cm] * 4,
        rowHeights=[0.7 * cm, 1.4 * cm]
    )
    table.setStyle(TableStyle([
        ('ALIGN',        (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('BACKGROUND',   (0, 0), (-1, -1), colors.HexColor('#f6f8fa')),
        ('BOX',          (0, 0), (0, -1), 0.5, BORDER),
        ('BOX',          (1, 0), (1, -1), 0.5, BORDER),
        ('BOX',          (2, 0), (2, -1), 0.5, BORDER),
        ('BOX',          (3, 0), (3, -1), 0.5, BORDER),
        ('TOPPADDING',   (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 6),
        ('LEFTPADDING',  (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    return table


# ─────────────────────────────────────────────
# ALERT ROW
# ─────────────────────────────────────────────

def alert_row_table(alert, styles):
    sev = alert['severity']
    bg  = severity_bg(sev)
    col = severity_color(sev)

    header = Table(
        [[
            Paragraph(f"#{alert['index']}", ParagraphStyle(
                'ai', fontName='Helvetica', fontSize=8, textColor=TEXT_MUTED
            )),
            Paragraph(sev, ParagraphStyle(
                'as', fontName='Helvetica-Bold', fontSize=9, textColor=col
            )),
            Paragraph(str(alert.get('original_label', 'N/A')), ParagraphStyle(
                'al', fontName='Helvetica', fontSize=8, textColor=TEXT_MAIN
            )),
            Paragraph(f"Port {alert.get('destination_port', '—')}", ParagraphStyle(
                'ap', fontName='Helvetica', fontSize=8, textColor=TEXT_MUTED
            )),
            Paragraph(
                f"H:{alert['confidence_high']}%  M:{alert['confidence_medium']}%  L:{alert['confidence_low']}%",
                ParagraphStyle('ac', fontName='Courier', fontSize=7, textColor=TEXT_MUTED)
            ),
        ]],
        colWidths=[1.2*cm, 2*cm, 4*cm, 2*cm, 4*cm],
        rowHeights=[0.6*cm]
    )
    header.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, -1), bg),
        ('LEFTPADDING',  (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING',   (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
        ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
        ('LINEBELOW',    (0, 0), (-1, -1), 0.3, BORDER),
    ]))

    reason_rows = []
    for r in alert.get('reasons', [])[:5]:
        direction = r.get('direction', '+')
        feat      = r.get('feature', '')
        val       = r.get('value', 0)
        shap_val  = r.get('shap', 0)
        dir_color = GREEN if direction == '+' else RED
        bar_width = min(int(abs(shap_val) * 3000), 80)

        reason_rows.append([
            Paragraph(direction, ParagraphStyle(
                'rd', fontName='Helvetica-Bold', fontSize=8, textColor=dir_color
            )),
            Paragraph(feat, ParagraphStyle(
                'rf', fontName='Courier', fontSize=7, textColor=TEXT_MAIN
            )),
            Paragraph(f"{val:,.3f}", ParagraphStyle(
                'rv', fontName='Courier', fontSize=7,
                textColor=TEXT_MUTED, alignment=TA_RIGHT
            )),
            Paragraph('|' * bar_width, ParagraphStyle(
                'rb', fontName='Courier', fontSize=6, textColor=dir_color
            )),
            Paragraph(f"{shap_val:+.4f}", ParagraphStyle(
                'rs', fontName='Courier', fontSize=7,
                textColor=dir_color, alignment=TA_RIGHT
            )),
        ])

    if reason_rows:
        reasons_table = Table(
            reason_rows,
            colWidths=[0.6*cm, 5.5*cm, 2.2*cm, 3.5*cm, 1.5*cm],
            rowHeights=[0.45*cm] * len(reason_rows)
        )
        reasons_table.setStyle(TableStyle([
            ('BACKGROUND',   (0, 0), (-1, -1), colors.HexColor('#f6f8fa')),
            ('LEFTPADDING',  (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING',   (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING',(0, 0), (-1, -1), 2),
            ('VALIGN',       (0, 0), (-1, -1), 'MIDDLE'),
            ('LINEBELOW',    (0, -1), (-1, -1), 0.5, BORDER),
        ]))
        return [header, reasons_table]

    return [header]


# ─────────────────────────────────────────────
# MAIN PDF BUILDER
# ─────────────────────────────────────────────

def generate_report(summary: dict, alerts: list) -> bytes:
    buffer = io.BytesIO()
    styles = build_styles()
    now    = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title='SOC Alert Triage Report',
        author='SOC Triage System',
    )

    story = []

    # ── Header ──
    story.append(Paragraph('SOC ALERT TRIAGE REPORT', styles['title']))
    story.append(Paragraph(f'Generated: {now}', styles['subtitle']))
    story.append(Paragraph(
        'Explainable AI-Based Network Intrusion Detection  |  CIC-IDS2017 Model',
        styles['subtitle']
    ))
    story.append(Spacer(1, 0.4*cm))
    story.append(section_rule())

    # ── Summary cards ──
    story.append(Paragraph('EXECUTIVE SUMMARY', styles['section']))
    story.append(summary_card_table(summary, styles))
    story.append(Spacer(1, 0.4*cm))

    total    = summary['total'] or 1
    high_pct = summary['high']   / total * 100
    med_pct  = summary['medium'] / total * 100
    low_pct  = summary['low']    / total * 100

    story.append(Paragraph(
        f"Analysis of <b>{summary['total']:,}</b> network flows completed. "
        f"<b><font color='#cf222e'>{summary['high']:,} ({high_pct:.1f}%)</font></b> flows classified HIGH — immediate attention required. "
        f"<b><font color='#9a6700'>{summary['medium']:,} ({med_pct:.1f}%)</font></b> classified MEDIUM — monitor and investigate. "
        f"<b><font color='#1a7f37'>{summary['low']:,} ({low_pct:.1f}%)</font></b> classified LOW — benign traffic.",
        styles['body']
    ))

    if summary.get('missing_features'):
        story.append(Spacer(1, 0.2*cm))
        mf = summary['missing_features']
        story.append(Paragraph(
            f"Note: {len(mf)} features absent from uploaded file and filled with zero: "
            f"{', '.join(mf[:5])}{'...' if len(mf) > 5 else ''}.",
            styles['small']
        ))

    story.append(Spacer(1, 0.4*cm))
    story.append(section_rule())

    # ── Severity breakdown ──
    story.append(Paragraph('SEVERITY BREAKDOWN', styles['section']))

    breakdown_data = [
        [
            Paragraph('Severity',         styles['label']),
            Paragraph('Count',            styles['label']),
            Paragraph('Percentage',       styles['label']),
            Paragraph('Action Required',  styles['label']),
        ],
        [
            Paragraph('HIGH', ParagraphStyle('h', fontName='Helvetica-Bold', fontSize=9, textColor=RED)),
            Paragraph(str(summary['high']),   styles['body']),
            Paragraph(f"{high_pct:.1f}%",     styles['body']),
            Paragraph('Immediate investigation required', styles['body']),
        ],
        [
            Paragraph('MEDIUM', ParagraphStyle('m', fontName='Helvetica-Bold', fontSize=9, textColor=YELLOW)),
            Paragraph(str(summary['medium']), styles['body']),
            Paragraph(f"{med_pct:.1f}%",      styles['body']),
            Paragraph('Investigate within 4 hours', styles['body']),
        ],
        [
            Paragraph('LOW', ParagraphStyle('l', fontName='Helvetica-Bold', fontSize=9, textColor=GREEN)),
            Paragraph(str(summary['low']),    styles['body']),
            Paragraph(f"{low_pct:.1f}%",      styles['body']),
            Paragraph('Log and review in daily summary', styles['body']),
        ],
    ]

    breakdown_table = Table(
        breakdown_data,
        colWidths=[2.5*cm, 2.5*cm, 3*cm, 9.2*cm],
    )
    breakdown_table.setStyle(TableStyle([
        ('BACKGROUND',     (0, 0), (-1, 0), colors.HexColor('#f6f8fa')),
        ('LINEBELOW',      (0, 0), (-1, 0), 0.5, BORDER),
        ('LINEBELOW',      (0, 1), (-1, -1), 0.3, colors.HexColor('#d0d7de')),
        ('LEFTPADDING',    (0, 0), (-1, -1), 8),
        ('RIGHTPADDING',   (0, 0), (-1, -1), 8),
        ('TOPPADDING',     (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING',  (0, 0), (-1, -1), 6),
        ('VALIGN',         (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [RED_BG, YELLOW_BG, GREEN_BG]),
    ]))
    story.append(breakdown_table)
    story.append(Spacer(1, 0.4*cm))
    story.append(section_rule())

    # ── Alert details ──
    story.append(Paragraph('ALERT DETAILS WITH SHAP EXPLANATIONS', styles['section']))
    story.append(Paragraph(
        'Each alert shows the top 5 features that drove the severity decision. '
        '+ indicates the feature pushed toward the predicted severity. '
        '- indicates it pushed away. Bar length represents relative SHAP magnitude.',
        styles['small']
    ))
    story.append(Spacer(1, 0.3*cm))

    # Column header
    col_header = Table(
        [[
            Paragraph('#',              styles['label']),
            Paragraph('SEVERITY',       styles['label']),
            Paragraph('ORIGINAL LABEL', styles['label']),
            Paragraph('DST PORT',       styles['label']),
            Paragraph('CONFIDENCE',     styles['label']),
        ]],
        colWidths=[1.2*cm, 2*cm, 4*cm, 2*cm, 4*cm],
        rowHeights=[0.5*cm]
    )
    col_header.setStyle(TableStyle([
        ('BACKGROUND',   (0, 0), (-1, -1), colors.HexColor('#f6f8fa')),
        ('LINEBELOW',    (0, 0), (-1, -1), 0.8, BORDER),
        ('LEFTPADDING',  (0, 0), (-1, -1), 6),
        ('TOPPADDING',   (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0, 0), (-1, -1), 4),
    ]))
    story.append(col_header)

    max_alerts = min(100, len(alerts))
    for alert in alerts[:max_alerts]:
        rows = alert_row_table(alert, styles)
        story.append(KeepTogether(rows))

    if len(alerts) > max_alerts:
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(
            f"Report shows first {max_alerts} of {len(alerts):,} alerts.",
            styles['small']
        ))

    # ── Footer ──
    story.append(Spacer(1, 0.6*cm))
    story.append(section_rule())
    story.append(Paragraph(
        f'SOC Alert Triage System  |  Explainable AI  |  '
        f'Model: Random Forest (CIC-IDS2017)  |  Generated: {now}',
        ParagraphStyle('footer', fontName='Helvetica', fontSize=7,
                       textColor=TEXT_MUTED, alignment=TA_CENTER)
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
