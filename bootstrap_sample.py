import zs
import os
import codecs
import time
import math
import numpy
import numpy as np
import pandas
import multiprocessing
import pdb


def get_mean_surp(bigrams_dict,zs_file_backward, word, cutoff): 
    start_time = time.time()    
    total_freq = 0
    surprisal_total = 0
    num_context = 0
    unweightedSurprisal = 0 
    searchTerm = word+u" " #need a trailing space
    #print 'Retrieving context probabilities for '+searchTerm   
    for record in zs_file_backward.search(prefix=searchTerm.encode('utf-8')):
        r_split = record.decode("utf-8").split(u"\t")
        ngram = r_split[0].split(u' ')
        #print r_split[0]
        count = int(r_split[1])
        if count > cutoff:
            total_freq += count
            context =u" ".join(ngram[1:][::-1])+u' '
            num_context += 1
            if context in bigrams_dict:
                total_context_freq = bigrams_dict[context]
                cond_prob = math.log(count / float(total_context_freq))
                #print cond_prob
                surprisal_total += (count * cond_prob) #this is weighted by the frequency of this context
                unweightedSurprisal +=  cond_prob #this is not
            else:
                pass
                #print('Missing context: '+ context) 
                #pdb.set_trace()
                #there should not be any missing values         
        else:
            continue    
    stop_time = time.time()
    st = None if total_freq == 0 else surprisal_total / float(total_freq)
    uwst = None if num_context == 0 else unweightedSurprisal / float(num_context)
    return (word, -1*st, -1*uwst, total_freq, num_context, (stop_time-start_time))


def get_trigram_surprisals_for_word(bigrams_dict,zs_file_backward, word):    
    start_time = time.time()    
    searchTerm = word+u" " #need a trailing space
    #print 'Retrieving context probabilities for '+searchTerm   

    ngrams = []    
    counts = [] 
    bigramCounts = [] 
    
    for record in zs_file_backward.search(prefix=searchTerm.encode('utf-8')):
        r_split = record.decode("utf-8").split(u"\t")
        ngram = r_split[0].split(u' ')        
        
        context =u" ".join(ngram[1:][::-1])+u' '
        if context in bigrams_dict:
            bigramCounts.append(bigrams_dict[context])
            ngrams.append(ngram)
            counts.append(np.float64(r_split[1]))
            
    if len(ngrams) == 0:
        pdb.set_trace()
        # why does this happen?
    
    print('get_trigram_surprisals_for_word elapsed:')
    print(time.time() - start_time)
            
    return(pandas.DataFrame({'ngram': ngrams,'trigramCount':counts, 'bigramCount':bigramCounts}))        

def getBootstrappedMeanInfoContent(bigrams_dict,zs_file_backward, word, sampleSize, bigram_total, intermediateSampleSize=None, original_sample_size=None, log=False, numSamples=1):
    sampleStore = []
    df = get_trigram_surprisals_for_word(bigrams_dict, zs_file_backward, word)      
    df['contextProb'] = df['bigramCount'] / bigram_total # this is the probability of the bigram for the record: the probability of each context in the dataset
    # summing it doesn't make sense 
    
    if not intermediateSampleSize: 
        ''' One stage downsampling, appropriate for a single language dataset'''

        prop = float(sampleSize) / original_sample_size        
        for sample_index in range(numSamples):
            # sample a new set of contexts. Note that this is sampling the number of times we see the context indepdendently for multiple ngrams that start with the same context
            df['newContextCount'] = [np.random.binomial(n=prop*bigram_total, p= df['contextProb'].values[i]) for i in range(df.shape[0])] # How often do we see this context? go from bigram total draws to prop* bigram_total
            
            df_nonzero = df[df['newContextCount'] > 0] # many of the above are zero
            df_nonzero['probContinuation'] = df_nonzero['trigramCount'] / df_nonzero['bigramCount']
            df_nonzero['newContinuationCount'] = [np.random.binomial(n=df_nonzero['newContextCount'].values[i], p= df_nonzero['probContinuation'].values[i]) for i in range(df_nonzero.shape[0])]

            #print(sum(df_nonzero['newContinuationCount']) / sum(df_nonzero['trigramCount']))
            rt = df_nonzero[df_nonzero['newContinuationCount'] > 0].sort_values(by='newContinuationCount', ascending=False)
            rt['newTrigramProbability'] = rt['newContinuationCount'] / rt['newContextCount']
            if np.sum(rt['newContinuationCount']) > 0:
                meanInfoContent = np.average(-1*np.log(rt['newTrigramProbability']), weights = rt['newContinuationCount']) 
            else:
                meanInfoContent = 0
            
            sampleDF = pandas.DataFrame({'mean_info_content':meanInfoContent, 'num_boot_contexts':rt.shape[0], 'boot_frequency':np.sum(rt['newContinuationCount'])}, index=[0])
            sampleDF['sample_index'] = sample_index
            sampleDF['propOfOriginalCorpus'] = prop
            sampleDF['sampleSize'] = sampleSize
            sampleDF['usesIntermediateSampleSize'] = False

            sampleStore.append(sampleDF)
        
        all_samples = pandas.concat(sampleStore)   
        return(all_samples)  
    else:           

        for sample_index in range(numSamples):         

            # Step 1: draw a sample of intermediate size. This is downsampling from the original full-size dataset to the intermediate sample size

            #sum(df['context_prob'])
            #why is this .78 for English 1T? b/c 78% of the bigrams in the dataset could be followed by "the"
            
            intermediateProp = (float(intermediateSampleSize)/original_sample_size)

            # draw new subset of contexts, x_w
            num_bigrams_to_draw_1 = np.round(intermediateProp * bigram_total)
            if log:
                print('intermediateSampleSize')
                print(intermediateSampleSize)
                print('original_sample_size')
                print(original_sample_size)
                print('bigram_total')
                print(bigram_total)
                print('num_bigrams_to_draw_1')
                print(num_bigrams_to_draw_1)            
            df['intermediateSample_ContextCount'] = [np.random.binomial(n=num_bigrams_to_draw_1, p= df['contextProb'].values[i]) for i in range(df.shape[0])]                             
            df['intermediateSample_ContextProb'] = df['intermediateSample_ContextCount'] / float(num_bigrams_to_draw_1)

            df_nonzero = df[df['intermediateSample_ContextCount'] > 0]
            df_nonzero['probContinuation'] = df_nonzero['trigramCount'] / df_nonzero['bigramCount']
            # draw new continuations for those contexts, d_w
            df_nonzero['intermediateSample_ContinuationCount'] = [np.random.binomial(n=df_nonzero['intermediateSample_ContextCount'].values[i], p= df_nonzero['probContinuation'].values[i]) for i in range(df_nonzero.shape[0])]            

            #print(sum(df_nonzero['newContinuationCount']) / sum(df_nonzero['trigramCount']))
            rt = df_nonzero[df_nonzero['intermediateSample_ContinuationCount'] > 0]
            rt['intermediateSample_TrigramProbability'] = rt['intermediateSample_ContinuationCount'] / rt['intermediateSample_ContextCount']                

            # Step 2: draw again, but using probabilities/counts from step 1
            # prop is computed wrt intermediate corpus
            
            prop = sampleSize / float(intermediateSampleSize)
            intermediate_bigram_total = num_bigrams_to_draw_1 # note this identity
            
            num_bigrams_to_draw_2 = np.round(prop * intermediate_bigram_total)
            rt['newContextCount'] = [np.random.binomial(n= num_bigrams_to_draw_2, p= rt['intermediateSample_ContextProb'].values[i]) for i in range(rt.shape[0])]        
            
            rt_nonzero = rt[rt['newContextCount'] > 0]
            rt_nonzero['newContinuationCount'] = [np.random.binomial(n=rt_nonzero['newContextCount'].values[i], p= rt_nonzero['intermediateSample_TrigramProbability'].values[i]) for i in range(rt_nonzero.shape[0])]
            
            rt_nonzero = rt_nonzero[rt_nonzero['newContinuationCount'] > 0].sort_values(by='newContinuationCount', ascending=False)                                         
            rt_nonzero['newTrigramProbability'] = rt_nonzero['newContinuationCount'] / rt_nonzero['newContextCount']
                                        
            
            if np.sum(rt_nonzero['newContinuationCount']) > 0:
                meanInfoContent = np.average(-1*np.log(rt_nonzero['newTrigramProbability']), weights = rt_nonzero['newContinuationCount']) 
            else:
                meanInfoContent = 0
            sampleDF = pandas.DataFrame({'mean_info_content':meanInfoContent, 'num_boot_contexts':rt_nonzero.shape[0], 'boot_frequency':np.sum(rt_nonzero['newContinuationCount'])}, index=[0])
            sampleDF['sample_index'] = sample_index
            sampleDF['propOfIntermediateCorpus'] = prop
            sampleDF['propOfOriginalCorpus'] = sampleSize / original_sample_size
            sampleDF['sampleSize'] = sampleSize
            sampleDF['usesIntermediateSampleSize'] = True
            sampleStore.append(sampleDF)
        all_samples = pandas.concat(sampleStore)    
        return(all_samples)  

def getTrajectoryForWord_wrapper(arg_dict):
    print(arg_dict)
    return(getTrajectoryForWord(**arg_dict))

def getTrajectoryForWord(word, sampleSizes, bigrams, backward_zs, bigram_total, intermediateSampleSize,  numSamples, original_sample_size=None, verbose=False):

    # print('Loading backwards-indexed counts')
    # backward_zs = zs.ZS(backward_zs_path)
    # print('Loading forwards-indexed counts')
    # bigrams, bigram_total = readForwardBigrams(forwards_txt_path)
    # print('Done loading datasets!')

    if verbose:
        print('Getting sampledd estimates for '+word)
    estimates = [getBootstrappedMeanInfoContent(bigrams, backward_zs, word, sampleSize, bigram_total, intermediateSampleSize, numSamples=numSamples, original_sample_size=original_sample_size) for sampleSize in sampleSizes]
    df = pandas.concat(estimates)    
    df['word'] = word    
    return(df)

def readForwardBigrams(forwards_txt_path):
    bigrams = {}
    bigram_total = 0
    f = codecs.open(forwards_txt_path, encoding='utf-8')
    for line in f:
        lineElements = line.split('\t')
        if len(lineElements) > 1:			
            key = lineElements[0]+u' ' 						
            val = int(lineElements[1])
            bigram_total += val
            bigrams[key] = val
        else:
            #print('broken bigram line: '+line) 
            #pdb.set_trace()
            pass
    return(bigrams, bigram_total)        

    
    
def getSampledSurprisalEstimates(result_path, dataset, language,  sampleSizes, intermediateSampleSize=None, log=False, numSamples=1):
    analysisPath = result_path + dataset+'/'+language+'/00_lexicalSurprisal/'
    surprisal_path = os.path.join(analysisPath, 'meanSurprisal.csv')
    print('Loading surprisal estimates for '+language+', '+dataset)
    surprisal = pandas.read_csv(surprisal_path, encoding='utf-8') #this is the new surprisal estimate
    dataset_token_count = np.sum(surprisal['frequency'])
    backwards_zs_path = os.path.join(analysisPath, '3gram-backwards.zs')
    forwards_txt_path = os.path.join(analysisPath,'2gram-forwards-collapsed.txt')
    
    # these will be loaded in the workers
    
    surprisal = surprisal.sort_values(by='frequency',ascending=False)
    #sample the wordset from across the frequency range
    #wordset = surprisal.iloc[np.arange(start=0, stop=25000, step=250)].word
    wordset = surprisal.iloc[np.arange(start=0, stop=25000, step=250)].word
    

    print('Loading backwards-indexed counts')
    backward_zs = zs.ZS(backwards_zs_path)
    print('Loading forwards-indexed counts')
    bigrams, bigram_total = readForwardBigrams(forwards_txt_path)
    print('Done loading datasets!')
    
    # # Multiprocessing -- can't share the ZS in any reasonable way
    # share an object across the processes? can't pickle the zs
    parallel_inputs = [{
         'word':word,
         'sampleSizes': sampleSizes,         
        'bigrams': bigrams,  # Take these from the global
        'backward_zs':backward_zs,
        'bigram_total':bigram_total,
        'intermediateSampleSize':intermediateSampleSize,
        'original_sample_size': dataset_token_count,
        'verbose': False,
        'numSamples':numSamples
    } for word in wordset]    

    # Multiprocessing approach -- doesn't work because you can't start a multiprocessing pool inside of a notebook
    # mp_pool = multiprocessing.pool.ThreadPool(8) # need to limit the number of workers because each has to load the zs separately. This might be slow because accessing the same disk contents
    # job_results = mp_pool.map(getTrajectoryForWord_wrapper, parallel_inputs)
    # mp_pool.close()
    # mp_pool.join()

    # Dumb parallelization
    #job_results = Parallel(n_jobs=8)(delayed(getTrajectoryForWord)(**i) for i in parallel_inputs)

    


    job_results = [getTrajectoryForWord(word, sampleSizes, bigrams, backward_zs, bigram_total, intermediateSampleSize, numSamples, original_sample_size=dataset_token_count) for word in wordset]
    
    all_df = pandas.concat(job_results)
    # Merge back in stats from surprisal to get the error estimates

    all_df = all_df.merge(surprisal)
            
    all_df['dataset'] = dataset
    all_df['language'] = language
    
    return(all_df)
    