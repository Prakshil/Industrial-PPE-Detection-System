"""
pdf_report.py — Generates formal OSHA 1910.132 & ISO 45001 safety compliance audit PDFs.
Uses reportlab to build executive-ready, printable landscape compliance ledgers.
"""

from __future__ import annotations

import io
import time
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_compliance_audit_pdf(
    violations: list[dict[str, Any]],
    zone_id: str | None = "all",
) -> bytes:
    """Builds an OSHA & ISO 45001 compliant PDF audit ledger report from violation records."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=landscape(letter),
        leftMargin=30,
        rightMargin=30,
        topMargin=26,
        bottomMargin=26,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#475569"),
    )
    kpi_number_style = ParagraphStyle(
        "KpiNumber",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=15,
        alignment=1,  # Center
        textColor=colors.HexColor("#0f172a"),
    )
    kpi_label_style = ParagraphStyle(
        "KpiLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=9,
        alignment=1,  # Center
        textColor=colors.HexColor("#64748b"),
    )
    cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1e293b"),
    )
    cell_mono = ParagraphStyle(
        "TableCellMono",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=6.5,
        leading=8.5,
        textColor=colors.HexColor("#475569"),
    )
    header_style = ParagraphStyle(
        "HeaderCell",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )

    story = []

    # 1. Header Banner
    export_time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    zone_str = (
        "All Zones (Plant-wide)"
        if not zone_id or zone_id == "all"
        else zone_id.replace("_", " ").upper()
    )

    header_data = [
        [
            Paragraph(
                "<b>OSHA 1910.132 &amp; ISO 45001 SAFETY COMPLIANCE AUDIT REPORT</b>", title_style
            ),
            Paragraph(
                f"<b>STATUS:</b> <font color='#16a34a'><b>ACTIVE MONITORING</b></font><br/><b>DATE:</b> {export_time_str} LOCAL",
                subtitle_style,
            ),
        ],
        [
            Paragraph(
                f"<b>FACILITY:</b> IPSIP Safety Command &bull; Sentinel Edge v2.0 &bull; <b>AUDIT SCOPE:</b> {zone_str}",
                subtitle_style,
            ),
            Paragraph(f"<b>VERIFICATION:</b> SHA-256 Cryptographic Chain", subtitle_style),
        ],
    ]
    header_table = Table(header_data, colWidths=[520, 212])
    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
            ]
        )
    )
    story.append(header_table)
    story.append(Spacer(1, 6))
    story.append(
        HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#1e3a8a"), spaceAfter=8)
    )

    # 2. Executive KPI Cards
    total_violations = len(violations)
    confirmed_count = sum(1 for v in violations if v.get("review_status") == "CONFIRMED")
    unreviewed_count = sum(
        1 for v in violations if v.get("review_status", "UNREVIEWED") == "UNREVIEWED"
    )
    dismissed_count = sum(
        1 for v in violations if v.get("review_status") in ("DISMISSED", "FALSE_POSITIVE")
    )
    compliance_score = max(0, 100 - (confirmed_count * 2))

    kpi_data = [
        [
            Paragraph(f"<font color='#0f172a'>{total_violations}</font>", kpi_number_style),
            Paragraph(f"<font color='#b91c1c'>{confirmed_count}</font>", kpi_number_style),
            Paragraph(f"<font color='#d97706'>{unreviewed_count}</font>", kpi_number_style),
            Paragraph(f"<font color='#2563eb'>{dismissed_count}</font>", kpi_number_style),
            Paragraph(f"<font color='#16a34a'>{compliance_score}%</font>", kpi_number_style),
        ],
        [
            Paragraph("TOTAL INCIDENTS LOGGED", kpi_label_style),
            Paragraph("CONFIRMED VIOLATIONS", kpi_label_style),
            Paragraph("PENDING TRIAGE / AUDIT", kpi_label_style),
            Paragraph("DISMISSED / TEST RUNS", kpi_label_style),
            Paragraph("SAFETY COMPLIANCE SCORE", kpi_label_style),
        ],
    ]
    kpi_table = Table(kpi_data, colWidths=[146, 146, 146, 146, 148])
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, 0), 4),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
                ("TOPPADDING", (0, 1), (-1, 1), 0),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 4),
            ]
        )
    )
    story.append(kpi_table)
    story.append(Spacer(1, 10))

    # 3. Incident Ledger Table
    table_headers = [
        Paragraph("ID", header_style),
        Paragraph("Timestamp (Local)", header_style),
        Paragraph("Zone ID", header_style),
        Paragraph("Worker", header_style),
        Paragraph("Missing PPE (Violations)", header_style),
        Paragraph("Verified PPE", header_style),
        Paragraph("Review Status", header_style),
        Paragraph("Reviewed By & Notes", header_style),
        Paragraph("Evidence Hash (SHA-256)", header_style),
    ]

    rows = [table_headers]

    for v in violations[:150]:  # Cap table at 150 rows per report for clean formatting
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(v["timestamp"]))
        missing_list = v.get("missing_ppe", [])
        if isinstance(missing_list, str):
            missing_list = [missing_list]
        present_list = v.get("present_ppe", [])
        if isinstance(present_list, str):
            present_list = [present_list]

        missing_str = ", ".join(missing_list).upper() if missing_list else "NONE"
        present_str = ", ".join(present_list).upper() if present_list else "NONE"
        status = v.get("review_status", "UNREVIEWED").upper()

        if status == "CONFIRMED":
            status_html = f"<font color='#991b1b'><b>CONFIRMED</b></font>"
        elif status == "UNREVIEWED":
            status_html = f"<font color='#d97706'><b>PENDING</b></font>"
        elif status == "FALSE_POSITIVE":
            status_html = f"<font color='#2563eb'><b>FALSE POS</b></font>"
        else:
            status_html = f"<font color='#475569'><b>{status}</b></font>"

        reviewer = v.get("reviewed_by") or "Automated Sentinel"
        notes = v.get("review_notes") or ""
        rev_text = f"<b>{reviewer}</b>"
        if notes:
            rev_text += f"<br/><font color='#64748b'>{notes[:40]}</font>"

        sha = v.get("evidence_sha256", "")
        sha_short = (sha[:14] + "...") if len(sha) > 16 else (sha or "N/A")

        rows.append(
            [
                Paragraph(f"<b>#{v.get('id', '')}</b>", cell_style),
                Paragraph(ts, cell_style),
                Paragraph(str(v.get("zone_id", "")).replace("_", " ").upper(), cell_style),
                Paragraph(f"Worker #{v.get('track_id', '')}", cell_style),
                Paragraph(f"<font color='#b91c1c'><b>{missing_str}</b></font>", cell_style),
                Paragraph(f"<font color='#16a34a'>{present_str}</font>", cell_style),
                Paragraph(status_html, cell_style),
                Paragraph(rev_text, cell_style),
                Paragraph(sha_short, cell_mono),
            ]
        )

    if len(rows) == 1:
        # No records found
        rows.append(
            [
                Paragraph("-", cell_style),
                Paragraph("No violations recorded in selected audit scope.", cell_style),
                Paragraph("-", cell_style),
                Paragraph("-", cell_style),
                Paragraph("Zero Violations (100% Compliant)", cell_style),
                Paragraph("-", cell_style),
                Paragraph("-", cell_style),
                Paragraph("-", cell_style),
                Paragraph("-", cell_mono),
            ]
        )

    # Landscape letter width: 792 pt. Margins: 30 each -> Available: 732 pt.
    # Col widths sum: 30 + 95 + 85 + 55 + 130 + 105 + 72 + 100 + 60 = 732 pt
    col_widths = [30, 95, 85, 55, 130, 105, 72, 100, 60]

    incident_table = Table(rows, colWidths=col_widths, repeatRows=1)
    incident_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(incident_table)
    story.append(Spacer(1, 14))

    # 4. Sign-off & Statutory Certification Block
    sign_data = [
        [
            Paragraph(
                "<b>EHS SAFETY OFFICER ATTESTATION</b>",
                ParagraphStyle(
                    "SignH",
                    parent=styles["Normal"],
                    fontSize=8,
                    fontName="Helvetica-Bold",
                    textColor=colors.HexColor("#0f172a"),
                ),
            ),
            Paragraph(
                "<b>PLANT OPERATIONS DIRECTOR SIGN-OFF</b>",
                ParagraphStyle(
                    "SignH2",
                    parent=styles["Normal"],
                    fontSize=8,
                    fontName="Helvetica-Bold",
                    textColor=colors.HexColor("#0f172a"),
                ),
            ),
        ],
        [
            Paragraph(
                "Signature: _____________________________________<br/>Date: ____________________  Badge ID: ___________",
                subtitle_style,
            ),
            Paragraph(
                "Signature: _____________________________________<br/>Date: ____________________  Badge ID: ___________",
                subtitle_style,
            ),
        ],
        [
            Paragraph(
                "<font size=7 color='#64748b'>Certified under OSHA 29 CFR 1910.132 Personal Protective Equipment standard &amp; ISO 45001:2018 Occupational Health and Safety Management Systems. Cryptographic evidence hashes verify continuous non-tampering audit trail.</font>",
                subtitle_style,
            ),
            Paragraph(
                "<font size=7 color='#64748b'>Report generated by Antigravity Sentinel Edge Computer Vision Platform.</font>",
                subtitle_style,
            ),
        ],
    ]
    sign_table = Table(sign_data, colWidths=[366, 366])
    sign_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEABOVE", (0, 0), (-1, 0), 1, colors.HexColor("#94a3b8")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    story.append(KeepTogether([sign_table]))

    doc.build(story)
    return buf.getvalue()
