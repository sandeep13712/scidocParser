# scidocParser
Tools:

A. Finding the quality of the research paper: This script uses tensorflow NLP toolkit to classify type of sentences in a research paper. Trained model is then exported to json and exported model is used to power our online paper parser available at https://www.sanmed.ca/pages/paperparser

B. Predicting research_area and nature of work from research paper.

1. Upload PDF
2. Text is retrived from PDF
3. Stopwords are removed
4. TF (term frequency is determined for biologically relevant keywords
5. Term frequency is being used to predict: research_area, techniques, work_nature (computational, integrative, experimental), relevant journals

Datasets: stopwords.txt, relevantkeywords.txt, journals.txt, research_areas.txt, techniques.txt 
