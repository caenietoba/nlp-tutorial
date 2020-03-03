import PyPDF2

my_file = open('US_Declaration.pdf', mode='rb')

pdf_reader = PyPDF2.PdfFileReader(my_file)

print(pdf_reader.numPages)

page_one = pdf_reader.getPage(0)

pdf_writer = PyPDF2.PdfFileWriter()

pdf_writer.addPage(page_one)

pdf_file = open('MY_BRAND_NEW.pdf', 'wb')

pdf_writer.write(pdf_file)

""" print(page_one.extractText()) """
pdf_file.close()

my_file.close()

pdf_file = open('MY_BRAND_NEW.pdf', 'rb')

pdf_reader = PyPDF2.PdfFileReader(pdf_file)

print(pdf_reader.getPage(0).extractText())

pdf_file.close()
