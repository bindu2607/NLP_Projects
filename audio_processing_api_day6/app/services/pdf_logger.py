"""
Comprehensive PDF logging service for audit trails and reporting.
"""
import os
from datetime import datetime
from typing import Dict, Any
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from app.core.config import get_settings

settings = get_settings()

class PDFLogger:
    """Advanced PDF logging with detailed reports and analytics."""

    def __init__(self):
        self.reports_dir = settings.REPORTS_DIR
        os.makedirs(self.reports_dir, exist_ok=True)
        self.styles = getSampleStyleSheet()
        # Custom styles
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            textColor=colors.darkblue
        ))

    async def log_transcription(self, filename: str, result: Dict[str, Any], user_id: str):
        """Log transcription request with detailed results."""
        report_data = {
            "type": "Transcription Report",
            "filename": filename,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "results": result
        }
        pdf_path = self._generate_pdf_path("transcription")
        self._create_transcription_pdf(pdf_path, report_data)
        return pdf_path

    async def log_voice_cloning(self, filename: str, text: str, result: Dict[str, Any], user_id: str):
        """Log voice cloning request with similarity analysis."""
        report_data = {
            "type": "Voice Cloning Report",
            "filename": filename,
            "text": text,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "results": result
        }
        pdf_path = self._generate_pdf_path("voice_cloning")
        self._create_voice_cloning_pdf(pdf_path, report_data)
        return pdf_path

    async def log_pipeline_processing(self, filename: str, results: Dict[str, Any], user_id: str):
        """Log complete pipeline processing with all stages."""
        report_data = {
            "type": "Pipeline Processing Report",
            "filename": filename,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "results": results
        }
        pdf_path = self._generate_pdf_path("pipeline")
        self._create_pipeline_pdf(pdf_path, report_data)
        return pdf_path

    def _generate_pdf_path(self, report_type: str) -> str:
        """Generate unique PDF file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_report_{timestamp}.pdf"
        return os.path.join(self.reports_dir, filename)

    def _create_transcription_pdf(self, pdf_path: str, data: Dict[str, Any]):
        """Create detailed transcription report."""
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []

        # Title and header
        story.append(Paragraph("Audio Transcription Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 12))

        # Basic information table
        basic_info = [
            ['Field', 'Value'],
            ['File Name', data['filename']],
            ['User ID', data['user_id']],
            ['Timestamp', data['timestamp']],
            ['Language Detected', data['results'].get('language', 'N/A')],
            ['Confidence', f"{data['results'].get('language_probability', 0):.2%}"],
            ['Duration', f"{data['results'].get('duration', 0):.2f}s"],
            ['Processing Time', f"{data['results'].get('processing_time', 0):.2f}s"]
        ]

        table = Table(basic_info)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(table)
        story.append(Spacer(1, 20))

        # Transcription text
        story.append(Paragraph("Transcribed Text:", self.styles['Heading2']))
        story.append(Paragraph(data['results'].get('text', 'No text available'), self.styles['Normal']))

        doc.build(story)

    def _create_voice_cloning_pdf(self, pdf_path: str, data: Dict[str, Any]):
        """Create detailed voice cloning report."""
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []

        # Title and header
        story.append(Paragraph("Voice Cloning Analysis Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 12))

        # Analysis results
        results = data['results']
        analysis_data = [
            ['Metric', 'Value', 'Assessment'],
            ['Similarity Score', f"{results.get('similarity_score', 0):.3f}", results.get('quality_rating', 'N/A')],
            ['Cloning Success', 'Yes' if results.get('cloning_successful', False) else 'No', ''],
            ['Processing Time', f"{results.get('processing_time', 0):.2f}s", ''],
            ['Audio Duration', f"{results.get('duration', 0):.2f}s", ''],
            ['Target Language', results.get('language', 'N/A'), '']
        ]

        table = Table(analysis_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(table)
        story.append(Spacer(1, 20))

        # Input text
        story.append(Paragraph("Synthesized Text:", self.styles['Heading2']))
        story.append(Paragraph(data['text'], self.styles['Normal']))

        doc.build(story)

    def _create_pipeline_pdf(self, pdf_path: str, data: Dict[str, Any]):
        """Create comprehensive pipeline processing report."""
        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        story = []

        # Title
        story.append(Paragraph("Complete Pipeline Processing Report", self.styles['CustomTitle']))
        story.append(Spacer(1, 12))

        # Pipeline summary
        results = data['results']
        metadata = results.get('metadata', {})

        summary_data = [
            ['Stage', 'Status', 'Details'],
            ['Audio Input', '✓ Complete', data['filename']],
            ['Transcription', '✓ Complete', f"Language: {metadata.get('source_language', 'N/A')}"],
            ['Translation', '✓ Complete', f"Target: {metadata.get('target_language', 'N/A')}"],
            ['Voice Cloning', '✓ Complete', f"Quality: {metadata.get('processing_quality', 'N/A')}"],
            ['Total Time', '', f"{results.get('total_processing_time', 0):.2f}s"]
        ]

        table = Table(summary_data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(table)
        doc.build(story)
