from datetime import datetime
from pathlib import Path

import xlwt
from xlrd import open_workbook
from xlutils.copy import copy


def output(filename, sheet, SvL, present):
    my_file = Path('excel/' + filename + str(datetime.now().date()) + '.xlsx')
    if my_file.is_file():
        rb = open_workbook('excel/' + filename + str(datetime.now().date()) + '.xlsx')
        book = copy(rb)
        sh = book.get_sheet(0)
        # file exists
    else:
        book = xlwt.Workbook()
        sh = book.add_sheet(sheet)
    style0 = xlwt.easyxf('font: name Times New Roman, color-index red, bold on',
                         num_format_str='#,##0.00')
    style1 = xlwt.easyxf(num_format_str='D-MMM-YY')

    sh.write(0, 0, datetime.now().date(), style1)

    col1_name = 'Name'
    col2_name = 'Email'
    col3_name = 'Gender'
    col4_name = 'Contact'
    col5_name = 'Present'
    sh.write(1, 0, col1_name, style0)
    sh.write(1, 1, col2_name, style0)
    sh.write(1, 2, col3_name, style0)
    sh.write(1, 3, col4_name, style0)
    sh.write(1, 4, col5_name, style0)
    for i,profile in enumerate(SvL):
        num, name, email, gender, contact = profile[:5]
        sh.write(i + 2, 0, name)
        sh.write(i + 2, 1, email)
        sh.write(i + 2, 2, gender)
        sh.write(i + 2, 3, contact)
        sh.write(i + 2, 4, present)

    fullname = filename + str(datetime.now().date()) + '.xls'
    book.save('excel/' + fullname)
    return fullname
