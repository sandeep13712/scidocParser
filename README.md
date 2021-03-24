# scidocParser

This repository contains tools to parse information from research paper. Following is the list of tools/libraries present in this repository:


Tools:
A. biology_vocabulary
Any machine model aimed at parsing biological research article essentially need keywords which are relevant and defines biological entities and phenomena. The purpose of this tool is to generate biological vocabulary from previously published paper. This vocabulary can then be used to develop NLP-based text analysis tools specifically for biology related papers.

B. article_context 
This package allows predicting research_area and work_type from research article. 

research_area could be one of the following:
biochemistry
biophysics
structural_biology
human_disease
immunology
microbiology
neurobiology
developmental_biology
cell_biology
cancer_biology
computational_biology
genetics

Work_type could be one the following:
computational
experimental
integrative

1. Upload PDF
2. Text is retrived from PDF
3. Stopwords are removed
4. TF (term frequency is determined for biologically relevant keywords
5. Term frequency is being used to predict: research_area, techniques, work_nature (computational, integrative, experimental), relevant journals

Datasets: stopwords.txt, relevantkeywords.txt, journals.txt, research_areas.txt, techniques.txt 

C. article_type 
This package allows predicting the type of article from its contents.

D. sentence_type
Given a sentence, this article predict the type of sentence. the sentence type could be one of the following:

Knownfact
Hypothesis
Conclusion
Result
Reasoning
Methodology

E. article_quality 
This package predict the quality score of an article based on its sentence composition and vocabulary richness and focus.



