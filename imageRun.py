import xlsxwriter as xs
import numpy as np
import re

workbook = xs.Workbook('hello.xlsx')

worksheet = workbook.add_worksheet()

worksheet.write('A1','Euclidian Distance')
for i in range(10):
    val = i + 2
    worksheet.write('A'+str(val), 3)

workbook.close()