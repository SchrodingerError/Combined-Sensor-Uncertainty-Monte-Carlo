import os
from fpdf import FPDF

from PIL import Image

class PDFGenerator(FPDF):
    def __init__(self):
        super().__init__()

        # Lists for storing items to memory
        self.stored_memory: list = []
        self.toc_entries: list = []  # List to store TOC entries

    def add_header(self, text, level: int = 1):
        """ Adds a header to the PDF. Level determines the size of the header. """
        size = {1: 18, 2: 16, 3: 14}.get(level, 18)
        self.set_font('DejaVu', 'B', size)
        self.multi_cell(0, 5, text)
        self.ln(2)  # Add a small line break after the header

    def add_text(self, text, bold: bool = False, indent: int = 0, newline=True):
        """ Adds normal text to the PDF """
        if bold == True:
            self.set_font('DejaVu', 'B', 8)
        else:
            self.set_font('DejaVu', '', 8)
        indents = " " * 4 * indent
        
        self.multi_cell(0, 5, indents + text)
        self.ln(1)

    def add_centered_text(self, text, bold: bool = False):
        """ Adds centered text to the PDF """
        if bold == True:
            self.set_font('DejaVu', 'B', 8)
        else:
            self.set_font('DejaVu', '', 8)
        self.multi_cell(0, 5, text, align='C')
        self.ln(1)

    def add_centered_image(self, image_path: str, width_ratio: float = 0.5, caption=None):
        """ Adds an image centered on the page, optionally with a caption below it. """
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        pdf_image_width = self.w * width_ratio
        aspect_ratio = image_height / image_width
        pdf_image_height = pdf_image_width * aspect_ratio
        x_position = (self.w - pdf_image_width) / 2

        # Center the image
        self.ln(1)
        self.image(image_path, x=x_position, w=pdf_image_width, h=pdf_image_height)

        # If a caption is provided, add it below the image
        if caption:
            self.set_font('DejaVu', 'I', 6)
            self.multi_cell(0, 10, caption, 0, 1, 'C')

    def format_value(self, value):
        """Format a number to 5 significant digits."""
        try:
            return f"{value:.5g}"
        except (ValueError, TypeError):
            return str(value)
    
    def create_metrics_table(self, title, metrics_list, column_labels):
        # Make title
        self.add_centered_text(title, bold=True)
        
        # Define row labels based on the keys of the first dictionary in the metrics list
        row_labels = list(metrics_list[0].keys())

        # Define column widths (you can adjust these as needed)
        col_widths = [40] + [13] * len(column_labels)
        cell_height = 10  # Adjust cell height as needed

        # Calculate the total width of the table
        table_width = sum(col_widths)

        # Define font settings
        self.set_font('DejaVu', 'B', 6)

        # Calculate the starting x position to center the table
        start_x = (210 - table_width) / 2  # Assuming A4 size paper (210mm width)

        # Set the starting x position
        self.set_x(start_x)

        # Add header row
        self.cell(col_widths[0], 10, 'Metrics/Sensor Noise Gain', 1, 0, 'C')
        for col_label in column_labels:
            self.cell(col_widths[1], cell_height, self.format_value(col_label), 1, 0, 'C')
        self.ln()

        # Add data rows
        self.set_font('DejaVu', '', 6)
        for row_label in row_labels:
            self.set_x(start_x)  # Reset the x position for each row
            x, y = self.get_x(), self.get_y()  # Save the current position
            self.set_font('DejaVu', 'B', 6)
            self.multi_cell(col_widths[0], cell_height, row_label, 1, 'C')
            self.set_font('DejaVu', '', 6)
            self.set_xy(x + col_widths[0], y)  # Adjust position for the next cells
            for metrics in metrics_list:
                self.cell(col_widths[1], cell_height, self.format_value(metrics[row_label]), 1, 0, 'C')
            self.ln()

    def make_pdf_from_memory(self, filepath: str) -> None:
        '''Makes the pdf from the stored list'''
        self.set_margins(5, 5, 5)  # left, top, and right margins in mm

        # Load a Unicode font
        current_directory = os.path.dirname(__file__)  # Get the directory where the script is located
        self.add_font('DejaVu', '', f"{current_directory}\\fonts\\dejavu-sans-condensed.ttf", uni=True)
        self.add_font('DejaVu', 'B', f"{current_directory}\\fonts\\dejavu-sans-condensedbold.ttf", uni=True)
        self.add_font('DejaVu', 'I', f"{current_directory}\\fonts\\dejavu-sans-condensedoblique.ttf", uni=True)
        self.set_font('DejaVu', '', 8)  # Set DejaVu as the default font


        self.add_header("Table of Contents")
        self.add_newline_memory()
        self.set_font('DejaVu', '', 8)
        for entry in self.toc_entries:
            self.cell(0, 10, f"{entry['text']}", 0, 0, 'L')
            self.cell(-20, 10, f"{entry['page']}", 0, 1, 'R')
        
        self.add_page()

        for item in self.stored_memory:
            match item["type"]:
                case "header":
                    self.add_header(item["text"], item["level"])
                case "text":
                    self.add_text(item["text"], item["bold"], item["indent"])
                case "centered_text":
                    self.add_centered_text(item["text"], item["bold"])
                case "centered_image":
                    self.add_centered_image(item["image_path"], item["width_ratio"], item["caption"])
                case "page":
                    self.add_page()
                case "newline":
                    self.ln(1)
                case "metrics_table":
                    self.create_metrics_table(item["title"], item["metrics_list"], item["column_labels"])

        self.output(filepath)

    def add_header_memory(self, text, level: int = 1, toc=True):
        dict_to_add = {
            "type": "header",
            "text": text,
            "level": level
        }
        self.stored_memory.append(dict_to_add)
        
        # Track header for TOC
        if toc:
            self.toc_entries.append({
                "text": text,
                "level": level,
                "page": sum(1 for item in self.stored_memory if item.get("type") == "page") + 2
            })
        

    def add_text_memory(self, text, bold: bool = False, indent: int = 0, newline: bool = True):
        dict_to_add = {
            "type": "text",
            "text": text,
            "bold": bold,
            "indent": indent,
            "newline": newline
        }
        self.stored_memory.append(dict_to_add)
        

    def add_centered_text_memory(self, text, bold: bool = False):
        dict_to_add = {
            "type": "centered_text",
            "text": text,
            "bold": bold
        }
        self.stored_memory.append(dict_to_add)
        

    def add_centered_image_memory(self, image_path: str, width_ratio: float = 0.5, caption=None):
        dict_to_add = {
            "type": "centered_image",
            "image_path": image_path,
            "width_ratio": width_ratio,
            "caption": caption
        }
        self.stored_memory.append(dict_to_add)
        
    def add_page_memory(self) -> None:
        dict_to_add = {
            "type": "page",
        }
        self.stored_memory.append(dict_to_add)

    def add_newline_memory(self) -> None:
        dict_to_add = {
            "type": "newline",
        }
        self.stored_memory.append(dict_to_add)

    def add_metrics_list_memory(self, metrics, indent=0) -> None:
        if "true_value" in metrics:
            self.add_text_memory(f"True System Output Value: {metrics['true_value']}", indent=indent, bold=True)
        if "median" in metrics:
            self.add_text_memory(f"Median: {metrics['median']}", indent=indent, bold=True)
        if "average" in metrics:
            self.add_text_memory(f"Average of Monte-Carlo: {metrics['average']}", indent=indent, bold=True)
        if "average_percent_difference" in metrics:
            self.add_text_memory(f"Average Percent Difference: {metrics['average_percent_difference']} %", indent=indent, bold=True)
        if "min_val" in metrics:
            self.add_text_memory(f"Minimum Value: {metrics['min_val']}", indent=indent)
        if "max_val" in metrics:
            self.add_text_memory(f"Maximum Value: {metrics['max_val']}", indent=indent)
        if "std_dev" in metrics:
            self.add_text_memory(f"Standard Deviation: ±{metrics['std_dev']}", indent=indent)
        if "std_dev_percent" in metrics:
            self.add_text_memory(f"Standard Deviation Percent: {metrics['std_dev_percent']} %", indent=indent)
        if "mean_absolute_error" in metrics:
            self.add_text_memory(f"Mean Absolute Error: {metrics['mean_absolute_error']}", indent=indent)
        if "mean_absolute_percent_error" in metrics:
            self.add_text_memory(f"Mean Absolute Percent Error: {metrics['mean_absolute_percent_error']} %", indent=indent)
        if "max_2std_error" in metrics:
            self.add_text_memory(f"Max Error 95% of the Time: {metrics['max_2std_error']}", indent=indent)
        if "max_2std_percent_error" in metrics:
            self.add_text_memory(f"Max Percent Error 95% of the Time: {metrics['max_2std_percent_error']} %", indent=indent, bold=True)
        if "max_3std_error" in metrics:
            self.add_text_memory(f"Max Error 99.73% of the Time: {metrics['max_3std_error']}", indent=indent)
        if "max_3std_percent_error" in metrics:
            self.add_text_memory(f"Max Percent Error 99.73% of the Time: {metrics['max_3std_percent_error']} %", indent=indent)



    def add_metrics_table_memory(self, title, metrics_list, column_labels) -> None:
        dict_to_add = {
            "type": "metrics_table",
            "title": title,
            "metrics_list": metrics_list,
            "column_labels": column_labels
        }
        self.stored_memory.append(dict_to_add)

