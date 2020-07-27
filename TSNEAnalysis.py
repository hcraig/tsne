import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import os

#Create empty lists
compiledtexts = []
text_authors = []
text_numbers = []
darwintext = []	
muirtext = []	
emersontext = []
thoreautext = []
edarwintext = []
marshtext = []
levisontext = []

#Function for reading in texts
def readintexts(author, listname):
	os.chdir('..')
	os.chdir(author)
	for file in os.listdir():
		author = open(file, 'r', encoding="latin1")
		text = author.readlines()	
		text = ' '.join(text)
		listname.append(text)
	
#Read in Charles Darwin texts
readintexts("darwin", darwintext)
text_authors.extend(['Charles Darwin'] * 18) 
text_numbers.extend([0] * 18)
del darwintext[0]

#Read in Muir texts
readintexts("muir", muirtext)
text_authors.extend(['Muir'] * 10)
text_numbers.extend([1] * 10)
del muirtext[0]

#Read in Emerson texts
readintexts("emerson", emersontext)
text_authors.extend(['Emerson'] * 7)
text_numbers.extend([2] * 7)
del emersontext[0]

#Read in Thoreau texts
readintexts("thoreau", thoreautext)
text_authors.extend(['Thoreau'] * 10)
text_numbers.extend([3] * 10)

#Read in Erasmus Darwin texts
readintexts("edarwin", edarwintext)
text_authors.extend(['Erasmus Darwin'] * 5)
text_numbers.extend([4] * 5)

#Read in Marsh texts
readintexts("marsh", marshtext)
text_authors.extend(['Marsh'] * 2)
text_numbers.extend([5] * 2)
del marshtext[0]

#Read in Levison texts
readintexts("levison", levisontext)
text_authors.extend(['Levison'] * 1)
text_numbers.extend([6] * 1)

#Combine all the texts into one list
compiledtexts.extend(darwintext)
compiledtexts.extend(muirtext)
compiledtexts.extend(emersontext)
compiledtexts.extend(thoreautext)
compiledtexts.extend(edarwintext)
compiledtexts.extend(marshtext)
compiledtexts.extend(levisontext)

#clean up the text
compiledtexts_final = [] 
for text in compiledtexts: 
	text = text.replace("\n",'')
	text = text.replace("\'",'')
	compiledtexts_final.append(text)   
 		
#Run TFIDF and vectorize
vec = TfidfVectorizer()
new = vec.fit_transform(compiledtexts)
vec.get_feature_names()[1]
final = TruncatedSVD(n_components=500, random_state=0).fit_transform(new)

#Run TSNE
finaltsne = TSNE(learning_rate=100).fit_transform(final)

#Function for creating plot with annotations
def createplot (tsneresults, fsize, annotate):
	a = tsneresults[:, 0]
	a = a.tolist()
	b = tsneresults[:, 1]
	b = b.tolist()
	n = text_numbers
	fig, ax = plt.subplots(figsize=(8,8))
	ax.scatter(a, b, c=text_numbers)
	for i, txt in enumerate(annotate):
 	   ax.annotate(txt, (a[i],b[i]), fontsize=fsize)
	plt.show()

#Function for creating plot without annotations
def createplot_noannotation (tsneresults):
	a = tsneresults[:, 0]
	a = a.tolist()
	b = tsneresults[:, 1]
	b = b.tolist()
	n = text_numbers
	fig, ax = plt.subplots(figsize=(4,4))
	ax.scatter(a, b, c=text_numbers)
	plt.show()

#Create plot with author name annotations
createplot(finaltsne, 5.5, text_authors)

#Read in text title doc
os.chdir('..')
titles = open('textnames.txt', 'r')

#Create list of all the text titles
texttitles = [] 
for line in titles: 
	texttitles.append(line) 
	
#Clean up the text titles list
texttitles_final = []
for title in texttitles:
	texttitles_final.append(title.strip())

for title in texttitles_final: 
	if title == "":
		texttitles_final.remove(title)

#Create plot with text title annotations
createplot(finaltsne, 5, texttitles_final)


#Tsne parameter tuning: Changing perplexity values
#Perplexity value 10
finaltsne = TSNE(perplexity=10.0).fit_transform(final)
createplot_noannotation(finaltsne)

#Perplexity value 40
finaltsne = TSNE(perplexity=40.0).fit_transform(final)
createplot_noannotation(finaltsne)

#Perplexity value 80
finaltsne = TSNE(perplexity=80.0).fit_transform(final)
createplot_noannotation(finaltsne)

#Perplexity value 120
finaltsne = TSNE(perplexity=120.0).fit_transform(final)
createplot_noannotation(finaltsne)

#Parameter tuning: Changing the learning rate
#Learning rate 10
finaltsne = TSNE(learning_rate=10).fit_transform(final)
createplot_noannotation(finaltsne)

#Learning rate 50
finaltsne = TSNE(learning_rate=50).fit_transform(final)
createplot_noannotation(finaltsne)

#Learning rate 100
finaltsne = TSNE(learning_rate=100).fit_transform(final)
createplot_noannotation(finaltsne)

#Learning rate 500
finaltsne = TSNE(learning_rate=500).fit_transform(final)
createplot_noannotation(finaltsne)


#Try TFIDF with stopwords excluded
vec = TfidfVectorizer(stop_words='english')
new = vec.fit_transform(compiledtexts)
vec.get_feature_names()[1]
final = TruncatedSVD(n_components=500, random_state=0).fit_transform(new)

createplot(finaltsne, 6, text_authors)


#Try different parameters for dimensionality reduction
#n_components = 50
final = TruncatedSVD(n_components=50, random_state=0).fit_transform(new)
finaltsne = TSNE(learning_rate=100).fit_transform(final)
createplot_noannotation(finaltsne)

#n_components = 100
final = TruncatedSVD(n_components=100, random_state=0).fit_transform(new)
finaltsne = TSNE(learning_rate=100).fit_transform(final)
createplot_noannotation(finaltsne)

#n_components = 1000
final = TruncatedSVD(n_components=1000, random_state=0).fit_transform(new)
finaltsne = TSNE(learning_rate=100).fit_transform(final)
createplot_noannotation(finaltsne)