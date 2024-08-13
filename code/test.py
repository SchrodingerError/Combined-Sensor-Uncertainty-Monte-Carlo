from fpdf import FPDF

class PDFWithTable(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Table Example', 0, 1, 'C')

    def add_table(self, data, col_widths, col_names):
        # Add a header row
        self.set_font('Arial', 'B', 12)
        for i, col_name in enumerate(col_names):
            self.cell(col_widths[i], 10, col_name, 1, 0, 'C')
        self.ln()

        # Add data rows
        self.set_font('Arial', '', 12)
        for row in data:
            for i, item in enumerate(row):
                self.cell(col_widths[i], 10, str(item), 1, 0, 'C')
            self.ln()

# Sample data
column_names = ['Name', 'Age', 'City']
data = [
    ['Alice', 30, 'New York'],
    ['Bob', 25, 'San Francisco'],
    ['Charlie', 35, 'Los Angeles']
]
column_widths = [40, 30, 50]  # Adjust column widths as needed

# Initialize PDF
pdf = PDFWithTable()
pdf.add_page()

# Add table to PDF
pdf.add_table(data, column_widths, column_names)

# Save PDF
pdf.output('table_example.pdf')
