# -*- coding: UTF-8 -*-
import re
import pandas
import os
import numpy as np
import scipy.stats.mstats
import unicodedata as ud


def clean_word(word, printResults=False):    
  originalWord = word
  word = word.replace(u'-',u'')  
  word = re.sub(u'\\(.*\\)',u'',word)  

  items = word.split(u' ')
  if len(items) > 1:
    if printResults:
    	print('Rejected: '+originalWord)
    return(None)
  else:
    return(re.sub(u' +','',items[0]).lower().strip())  



ids_to_language = {
    'ces':u'Czech',
    'nld':u'Dutch', 
    'eng':u'English',
    'fra':u'French',
    'deu':u'German', 
    'ita':u'Italian',
    'pol':u'Polish',
    'por':u'Portuguese',
    'ron':u'Romanian',
    'rus':u'Russian',
    'spa':u'Spanish',
    'swe':'Swedish',
    'heb':'Hebrew'
}

ids_to_gb12 = {
    'eng':'eng-all',
    'spa':'spa-all',
    'deu':'ger-all',
    'rus':'rus-all', 
    'heb':'heb-all',
    'fra':'fre-all'
}

gb12_to_ids = {v: k for k, v in ids_to_gb12.items()}

ids_to_g1t = {
    'eng':'ENGLISH',
    'spa':'SPANISH',
    'deu':'GERMAN',
    'fra':'FRENCH',
    'ces':'CZECH',
    'swe':'SWEDISH',        
    'pol':'POLISH', 
    'ron':'ROMANIAN',
    'ita':'ITALIAN',
    'por':'PORTUGUESE',
    'nld':'DUTCH' 
}               
               
gb1t_to_ids = {v: k for k, v in ids_to_g1t.items()}
               
ids_to_opus = {
    'eng':'en',
    'spa':'es',
    'deu':'de',
    'fra':'fr',
    'ces':'cs',
    'swe':'sv',        
    'pol':'pl', 
    'ron':'ro',
    'ita':'it',
    'por':'pt',
    'nld':'nl',
    'rus':'ru', 
    'heb':'he',
}   

datasetNameToShort = {
    'GoogleBooks2012':'GB12',
    'Google1T':'G1T',
    'OPUS':'OPUS',
    'BNC':'BNC',
}
    
               
opus_to_ids = {v: k for k, v in ids_to_opus.items()}

ids_translator = {'OPUS':ids_to_opus, 'GoogleBooks2012':ids_to_gb12, 'Google1T':ids_to_g1t}


def retrieveMeasuresForIDSitems(analysis_path, wordlist, conceptlist, languages):
	rdf_list = [] 
	for language in languages:
		for dataset in ['OPUS']:
			if language in ids_translator[dataset]:
				wordDF = pandas.DataFrame({'word': [ud.normalize('NFC', x) for x in wordlist[language]]})
				# contain \u[...] -> contain \x[...]

				lex = pandas.read_csv(os.path.join(analysis_path,dataset, ids_translator[dataset][language],'00_lexicalSurprisal','opus_meanSurprisal.csv'), encoding='utf-8')

				lex['unigramSurprisal'] = -1 * np.log(lex['frequency'] / sum(lex['frequency']))    
				def normalizeGracefully(x):
					try:
						return(ud.normalize('NFC', x))
					except:
						return('NOT NORMALIZABLE')
				lex['word'] = [normalizeGracefully(x) for x in lex['word']]

				# contain \x[...]

				sublex = pandas.read_csv(os.path.join(analysis_path,dataset, ids_translator[dataset][language],'01_sublexicalSurprisal','25000_sublex.csv'), encoding='utf-8')
				sublex['word'] = [normalizeGracefully(x) for x in sublex['word']]
				lex = lex.merge(sublex, on='word')

				if language == 'rus':					

					# need to convert lex to cyrillic

					cyrillic_map = pandas.read_csv('IDS/cyrillic_map.csv', encoding='utf-8')
					cyrillic_map['lower_cyr'] = [ud.normalize('NFC', x) for x in cyrillic_map['lower_cyr']]
					cyr_to_scholarly = dict(zip(cyrillic_map['lower_cyr'], cyrillic_map['scholarly']))
					
					def convertWord(word, charmap):
						try:
							rstr = ''.join([charmap[ud.normalize('NFC', x)] for x in list(word)])
							rstr = rstr.replace(u'ʹ',u'')
							#print(word + u' '+ rstr)
							return(rstr)
						except:
							print('FAILED TO CONVERT')							
							print(word)
							print(list(word))
							return('FAILED TO CONVERT')

					lex['word'] = [convertWord(x, cyr_to_scholarly) for x in lex['word']]

				if language == 'deu':
					# German nouns are uppercase!
					wordDF.word = [x.lower() for x in wordDF.word]
					

				rdf = wordDF.merge(lex, on='word')

					
				concept_df = conceptlist[language]
				concept_df.word = [ud.normalize('NFC', x) for x in concept_df.word]

				rdf = rdf.merge(conceptlist[language], on='word')
				rdf['language'] = language
				rdf['dataset'] = dataset


				if 'ipa_ss' in rdf.columns:
					rdf = rdf.sort_values(by=['ipa_ss'], ascending=False)
					rdf[u'ipa_ss_rank'] = np.arange(rdf.shape[0]) 
					rdf = rdf.sort_values(by=[u'ipa_n'], ascending=False)
					rdf[u'ipa_n_rank'] = np.arange(rdf.shape[0]) 
					rdf['ipa_ss_z'] = scipy.stats.mstats.zscore(rdf['ipa_ss'])
					rdf['ipa_n_z']  = scipy.stats.mstats.zscore(rdf['ipa_n'])


				rdf = rdf.sort_values(by=['character_ss'], ascending=False)
				rdf[u'character_ss_rank'] = np.arange(rdf.shape[0]) 
				rdf = rdf.sort_values(by=[u'character_n'], ascending=False)
				rdf[u'character_n_rank'] = np.arange(rdf.shape[0])     
				    
				rdf = rdf.sort_values(by=[u'unigramSurprisal'], ascending=False)
				rdf['unigramSurprisalRank'] = np.arange(rdf.shape[0])     
				rdf['unigramSurprisal_z'] = scipy.stats.mstats.zscore(rdf['unigramSurprisal'])
				rdf['mean_surprisal_weighted_z'] = scipy.stats.mstats.zscore(rdf['mean_surprisal_weighted'])
				rdf['character_ss_z'] = scipy.stats.mstats.zscore(rdf['character_ss'])
				rdf['character_n_z']  = scipy.stats.mstats.zscore(rdf['character_n'])

				rdf_list.append(rdf)                  
	
	all_rdf = pandas.concat(rdf_list)    
	return(all_rdf)

