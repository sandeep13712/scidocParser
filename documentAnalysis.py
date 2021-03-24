import pandas as pd
import numpy as np

import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
import io

def pdfparser(data):
    fp = open(data, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data =  retstr.getvalue()
    print(data)



#load the data
fullDataSets = pd.read_csv('dataForDocumentParsing.csv',sep=',')

# break data into train, validation and test subsets
trainingRatio = 0.6 
validateRatio = 0.2
testRatio = 0.2
trainingSize = int(trainingRatio*len(fullDataSets.index))
validateSize = int(validateRatio*len(fullDataSets))
testSize = int(testRatio*len(fullDataSets))
trainingDataSet =  fullDataSets.iloc[range(0,trainingSize)]
validateDataSet =  fullDataSets.iloc[trainingSize+testSize:]
testDataSet =  fullDataSets.iloc[trainingSize:trainingSize+testSize]

# parse PDF and create feature matrix

for index,pdfFile in trainingDataSet.iterrows():
    print(pdfFile['pdfId'])
    pdfparser(sys.argv[1])  




# print(trainingDataSet)
# print(validateDataSet)
# print(testDataSet)