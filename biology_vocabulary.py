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

valid_research_areas = "\n".join([
' ',
'1. biochemistry',
'2. biophysics',
'3. structural_biology',
'4. human_disease',
'5. immunology',
'6. microbiology',
'7. neurobiology',
'8. developmental_biology',
'9. cell_biology',
'10. cancer_biology',
'11. computational_biology',
'12. genetics',
'   ',    
])

valid_work_type = "\n".join([
' ',
'1. computational',
'2. experimental',
'3. hybrid/integrative',
'4. data-driven (population study, clinical)',
'5. field_study (e.g., in ecological systems)',
'   ',    
])

valid_article_type = "\n".join([
' ',
'1. research_article',
'2. review_article',
'3. short_article',
'4. perspective, opinion, and commentary',
'5. book_chapter',
'6. news_and_views',
'7. clinical_case_study',
'8. clinical_trial', 
'   ',    
])

def pdfparser(data):
    fp = open(data, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data =  retstr.getvalue()
    data = re.sub(r'\d+|\W+|\b\w\b',' ',data)
    data = re.sub(r'\W+',' ',data)
    data = re.sub(r'\s+',' ',data)
    return data

def createVocabulary():
    filesToParse = glob.glob('../pdfs/*.pdf')
    print(filesToParse)
    documentCorpus = list()

    metaData = ""

    for fileName in filesToParse[0:3]:
        try:
            print(fileName)
            data = pdfparser(fileName) 
            documentCorpus.append(data)
            print('total words: '+str(len(data.split())))

            researchArea = input('select research area (comma separated) : '+valid_research_areas+"\n(E.g., 1,3,4): ")
            work_type = input('select work type : '+valid_work_type+"\n(E.g., 1): ")
            article_type = input('select article type : '+valid_article_type+"\n(E.g., 1): ")
            metaData +=  '{0},{1},{2},{3}\n'.format(len(data.split()),researchArea.replace(',',' '),work_type.replace(',',' '),article_type.replace(',',' '))
            print(metaData)
            
        except:
            print('can not process: '+fileName)
        # print(data)

    count_vect = CountVectorizer() #analyzer='word', token_pattern=r'\w{1,}')
    fatureVector = count_vect.fit_transform(documentCorpus)

    with open('metaData.txt', 'w+') as f:
        f.write(metaData)

    with open('featureName.txt', 'w+') as f:
        featureNames = ",".join(count_vect.get_feature_names())
        f.write(featureNames)

    # print(fatureVector)

    wordFrequency = ''
    for x in fatureVector:
            for ind,freq in zip(x.indices,x.data):
                wordFrequency += '{0},{1}\n'.format(ind, freq)
                # print('{0}->{1}'.format(ind, freq))
    

    with open('featureWeights.txt', 'w+') as f:
        f.write(wordFrequency)
        

createVocabulary()
exit(0)