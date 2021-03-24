import pandas as pd
import numpy as np

import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
import io
import re

import glob

from sklearn.feature_extraction.text import CountVectorizer
    
def createFeatureVector(corpusList):
    count_vect = CountVectorizer() #analyzer='word', token_pattern=r'\w{1,}')
    print(corpusList)
    fatureVector = count_vect.fit_transform(corpusList)
    print(count_vect.get_feature_names())
    print(fatureVector.toarray())
    print(fatureVector.toarray().shape)
    return fatureVector

def pdfparser(data):
    fp = open(data, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data =  retstr.getvalue()
    data = re.sub(r'\d+|\W+|\b\w\b',' ',data)
    data = re.sub(r'\W+',' ',data)
    # data = re.sub(r'\b\w\b', '', data) #remove single character word
    data = re.sub(r'\s+',' ',data)
    # createFeatureVector(data)
    return data

def createVocabulary():
    filesToParse = glob.glob('pdfs/*.pdf')
    print(filesToParse)

    documentCorpus = list()
    for fileName in filesToParse:

        try:
            print(fileName)
            data = pdfparser(fileName) 
            documentCorpus.append(data)
            print('total words: '+str(len(data.split())))
        except:
            print('can not process: '+fileName)
        # print(data)
    createFeatureVector(documentCorpus)


createVocabulary()
exit(0)


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

#parse PDF and create feature matrix

for index,pdfFile in trainingDataSet.iterrows():
    print(pdfFile['pdfId'])
    pdfparser('SciReport.pdf')  
    break




# print(trainingDataSet)
# print(validateDataSet)
# print(testDataSet)