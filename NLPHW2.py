from collections import Counter, namedtuple
from math import log
import nltk
import sys  
import string
reload(sys)  
sys.setdefaultencoding('utf8')


LanguageModel = namedtuple('LanguageModel', 'num_tokens, vocab, updateNminus1grams, updateNgram, D') # holds counts for the lm
DELIM = "_" # delimiter for tokens in an ngram

def tokenize_text(text):
	""" Converts a string to a list of tokens """
	tokens = []
	tokens.extend(nltk.word_tokenize(text))
	return tokens

def generate_ngrams(tokens, n):
	""" Returns a list of ngrams made from a list of tokens """
	ngrams = []
	if n > 0:
		for i in range(0, len(tokens)-n+1):
			ngrams.append(DELIM.join(tokens[i:i+n]))
	return ngrams

def build_lm(text, n):
	""" Builds an ngram language model. """
	tokens = tokenize_text(text)
	num_tokens = len(tokens)
	vocab = set(tokens)
	nminus1grams = Counter(generate_ngrams(tokens, n - 1))
	ngrams = Counter(generate_ngrams(tokens, n))
	updateNgram = Unknownword(ngrams)
	updateNminus1grams = Unknownword(nminus1grams)
	D = calculateD(updateNgram, updateNminus1grams)
	return LanguageModel(num_tokens, vocab, updateNminus1grams, updateNgram, D)

#used to replace all of the unknown word with the unknown ngram
def Unknownword(ngram):
	UnknownwordList = []
	UnknownwordList = {k:v for k,v in Counter.iteritems(ngram) if v <= 1}
	ngram = {k:v for k,v in Counter.iteritems(ngram) if v > 1}
	ngram2 = {k:v for k,v in Counter.iteritems(ngram) if v == 2}
	n1Count =len(UnknownwordList)
	ngram.update({'Unknown':n1Count}) 
	return ngram

def katz_prob(lm, token, history = None):
	D = float(lm.D)
	if history == None:
		ngram_count_l = lm.updateNgram.get(token, 0)
		prefix_count_l = lm.num_tokens
		#print float(ngram_count_l)
		#print float(prefix_count_l)
		return log(float(float(ngram_count_l) / float(prefix_count_l)))
	elif lm.updateNgram.get(history+DELIM+token, 0) > 0: 
		#print lm.updateNgram.get(history+DELIM+token, 0)
		#print lm.updateNminus1grams.get(history, 0)
		ngram_count_l = lm.updateNgram.get(history+DELIM+token, 0)-D
		prefix_count_l = lm.updateNminus1grams.get(history, 0)
		#print float(ngram_count_l / prefix_count_l)
		#print log(float(ngram_count_l / prefix_count_l))
		return log(float(ngram_count_l / prefix_count_l))
	elif lm.updateNgram.get(history+DELIM+token, 0) == 0: 
		#print alpha(history, lm, D)*lm.updateNminus1grams.get(history, 0)/lm.num_tokens
		#print log(alpha(history, lm, D)*lm.updateNminus1grams.get(history, 0)/lm.num_tokens)
		return log(alpha(history, lm, D)*lm.updateNminus1grams.get(history, 0)/lm.num_tokens)

#calculate D in absolute discount
def calculateD(ngram, nminus1grams):
	return ngram.get('Unknown')/(ngram.get('Unknown')+2*len({k:v for k,v in Counter.iteritems(ngram) if v == 2}))

#calculate alpha
def alpha(given, lm, d):
	sum_exists = 0.0
	sum_unigram = 0.0
	for e in lm.updateNminus1grams.keys():
		if lm.updateNgram.get(given+e, 0) != 0:
			#print "sum_unigram"
			#print sum_exists
			sum_exists += float((lm.updateNgram.get(e)-d) / lm.updateNminus1grams.get(given))
		else:
			sum_unigram += float(lm.updateNminus1grams.get(e))	
	return float((1-sum_exists)/sum_unigram/lm.num_tokens)

if __name__ == '__main__':
	""" example usage"""

	rText = ""
	pText = ""
	countR = 0.0
	countP = 0.0
	with open("hw2_train.txt", 'r') as file:
		for line in file:
			if line.startswith('r: '):
				rText = rText+line.split('r: ', 2)[1]
			else:
				pText = pText+line.split('p: ', 2)[1]
	#use for bigram
	n= 2
	#use for unigram and bayesian
	#n =
	#use for baysian model
	rprobability= log(float(500.0/2000.0))
	pprobability= log(float(1500.0/2000.0))

	lmR = build_lm(rText, n) #bigram model
	lmP = build_lm(pText, n) 
	'''
	used to print the the token of unknowns in the chart
	print lmR.num_tokens
	print lmR.updateNgram.get('Unknown')
	print lmP.num_tokens
	print lmP.updateNgram.get('Unknown')
	print "probability of unknown in Review"
	print float(float(lmR.updateNgram.get('Unknown'))/float(lmR.num_tokens))
	print "probability of unknown in Plot"
	print float(float(lmP.updateNgram.get('Unknown'))/float(lmP.num_tokens))
	'''
	correctpostive = 0
	falsepositive = 0
	correctnegative = 0
	falsenegative = 0
	with open("hw2_test.txt", 'r') as file:
		for line in file:
			if line.startswith('r: '):
				countR += 1
				rCorrect = True 
			else:
				countP += 1
				rCorrect = False
			history = line[3:]		
			#processing Text for the plot dictionary
			testingtext = []
			testingtext.extend(nltk.word_tokenize(history))
			if n >1:
				for l in range(len(testingtext)):
					if testingtext[l] not in lmP.updateNminus1grams.keys():
						testingtext[l] = ('Unknown')
			else:
				for l in range(len(testingtext)):
					if testingtext[l] not in lmP.updateNgram.keys():
						testingtext[l] = ('Unknown')
			probs = 0.0	

			for i in range(len(testingtext)):
				if n >1:
					if i < n-2:
						probs += katz_prob(lmP, testingtext[i], None)
					else:
						probs += katz_prob(lmP, testingtext[i], testingtext[i-1])
				else:
					probs += katz_prob(lmP, testingtext[i], None)

			#processing Text for the plot dictionary
			testingtextR = []
			testingtextR.extend(nltk.word_tokenize(history))
			if n >1:
				for l in range(len(testingtextR)):
					if testingtextR[l] not in lmR.updateNminus1grams.keys():
						testingtextR[l] = ('Unknown')
			else:
				for l in range(len(testingtextR)):
					if testingtextR[l] not in lmR.updateNgram.keys():
						testingtextR[l] = ('Unknown')
						
			probsR = 0.0

			for j in range(len(testingtextR)):
				if n >1:
					if i < n-2:
						probsR += katz_prob(lmR, testingtextR[j], None)
					else:
						probsR += katz_prob(lmR, testingtextR[j], testingtextR[j-1])
				else:
					probsR += katz_prob(lmR, testingtextR[j], None)
			#print probs
			#print probsR
			# use below for Bayesian probability
			if (probs-pprobability) > (probsR-rprobability):
			#if probs > probsR:
				#print "Plot"
				if not rCorrect:
					correctnegative+=1
				else:
					falsenegative+=1
			else: 
				#print "Review"
				if rCorrect:
					correctpostive+=1
				else:
					falsepositive+=1
	precision = float(correctpostive/float(correctpostive+falsepositive))
	recall = float(correctpostive/float(correctpostive+falsenegative))
	accuracy = float((correctpostive+correctnegative)/float(correctpostive+falsenegative+correctnegative+falsepositive))
	print "correctpostive"
	print correctpostive
	print "falsepositive"
	print falsepositive
	print "correctnegative"
	print correctnegative
	print "falsenegative"
	print falsenegative
	print "accuracy"
	print accuracy
	print "precision:"
	print float(correctpostive/float(correctpostive+falsepositive))
	print "recall:"
	print float(correctpostive/float(correctpostive+falsenegative))
	print "F1:"
	print float(precision*recall*2/(precision+recall))

