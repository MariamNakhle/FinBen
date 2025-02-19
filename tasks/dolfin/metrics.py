import evaluate
from comet import download_model, load_from_checkpoint
import pysbd
import re
import string
import sys


model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

custom_separator = '_*dolfin-sep*_'

#### BLEU ###
def bleu(references, predictions):
    # TODO: now references also contain sources
    return (references[0].split(custom_separator)[1], predictions[0])

def agg_bleu(items):
    bleu_fn = evaluate.load("bleu")
    references, predictions = zip(*items)
    return bleu_fn.compute(predictions=predictions, references=references)["bleu"]


#### COMET ###
def comet(references, predictions, batch_size=8):
    return (references[0], predictions[0], batch_size)

def agg_comet(items):
    references, predictions, batch_size = zip(*items)

    batch_size = batch_size[0]
    data = [{'src': ref.split(custom_separator)[0], 'mt': mt, 'ref': ref.split(custom_separator)[1]} for mt, ref in zip(predictions, references)]
    scores = comet_model.predict(data, batch_size=batch_size)
    return scores['system_score']


#### COMET SLIDE ###
def comet_slide(references, predictions, batch_size=8):
    return (references[0], predictions[0], batch_size)

def agg_comet_slide(items):
    references, predictions, batch_size = zip(*items)
    segmenters = init_segmenters()

    sources = [x.split(custom_separator)[0] for x in references]
    targets = [x.split(custom_separator)[1] for x in references]
    src_langs = [x.split(custom_separator)[2] for x in references]
    trg_langs = [x.split(custom_separator)[3] for x in references]
    batch_size = batch_size[0]

    data_lists, mapping = sliding_window(sources=sources, translations=predictions, references=targets, segmenters=segmenters, src_langs=src_langs, trg_langs=trg_langs)

    flat_data = [item for sublist in data_lists for item in sublist]
    flat_mapping = [item for sublist in mapping for item in sublist]

    scores = comet_model.predict(flat_data, batch_size=batch_size, gpus=1)
    # TODO: is flat data really flat???

    individual_scores = scores['scores']

    segment_level_scores = []
    for seg_id in sorted(set(flat_mapping)): # for every unique id
        scores_for_id = [score for i, score in enumerate(individual_scores) if flat_mapping[i] == seg_id]
        avg = sum(scores_for_id) / len(scores_for_id)
        segment_level_scores.append(avg)

    return sum(segment_level_scores) / len(segment_level_scores)


def init_segmenters():
    seg_en = pysbd.Segmenter(language="en", clean=False)
    seg_fr = pysbd.Segmenter(language="fr", clean=False)
    seg_es = pysbd.Segmenter(language="es", clean=False)
    seg_it = pysbd.Segmenter(language="it", clean=False)
    seg_de = pysbd.Segmenter(language="de", clean=False)

    segmenters = {
    'seg_en' : seg_en,
    'seg_fr':seg_fr,
    'seg_es':seg_es,
    'seg_it':seg_it,
    'seg_de':seg_de
    }
    return segmenters

def sliding_window(sources, translations, references, segmenters, src_langs, trg_langs, window_size=3, stride=1):
    assert len(sources) == len(translations) == len(references), "Not the same number of segments"

    print('========== SLIDING STARTS HERE =========')
    counter = -1 # counter per segments
    data_lists = []
    mapping=[]
    for src, hyp, ref, src_lang, trg_lang in zip(sources, translations, references, src_langs, trg_langs):
        print('===> New segment')
        print('src')
        print(src)
        print('hyp')
        print(hyp)
        print('ref')
        print(ref)
        
        seg_src=segmenters['seg_'+src_lang]
        seg_trg=segmenters['seg_'+trg_lang]

        src = remove_markdown(src)
        hyp = remove_markdown(hyp)
        ref = remove_markdown(ref)
        
        counter+=1
        _segmented_src = seg_src.segment(src) # turn the section into sentences
        _segmented_hyp = seg_trg.segment(hyp)
        _segmented_ref = seg_trg.segment(ref)

        segmented_src = [x for x in _segmented_src if not is_only_punctuation(x.replace(' ', '').strip())] # remove sentences that are just punctuation
        segmented_hyp = [x for x in _segmented_hyp if not is_only_punctuation(x.replace(' ', '').strip())]
        segmented_ref = [x for x in _segmented_ref if not is_only_punctuation(x.replace(' ', '').strip())]

        # section shorter that window size, no need to slide
        if all(len(lst) < window_size for lst in [segmented_src, segmented_hyp, segmented_ref]):
            print('Shorter than window, no sliding needed')
            data = shorter_than_window(segmented_src, segmented_hyp, segmented_ref)
            
        # if we need to slide
        else:
            print('Longer than window, lets do dome sliding')
            # if source, target and ref do not have the same length, we will rearrange the sentences
            if not(len(segmented_src) == len(segmented_hyp) == len(segmented_ref)):
                print('Not the same length between src, ref, trg')
                shortest = min(len(segmented_src), len(segmented_hyp), len(segmented_ref))
                
                segmented_src = shorten_if_needed(segmented_src, shortest)
                segmented_hyp = shorten_if_needed(segmented_hyp, shortest)
                segmented_ref = shorten_if_needed(segmented_ref, shortest)
                
            # check if by shortening they got shorter than the window size
            if all(len(lst) < window_size for lst in [segmented_src, segmented_hyp, segmented_ref]):
                print('Got shorter than window')
                data = shorter_than_window(segmented_src, segmented_hyp, segmented_ref)
            else:
                print('Longer than window, lets do dome sliding')
                data =  bigger_than_window(segmented_src, segmented_hyp, segmented_ref, window_size, stride)
        print(data)
        data_lists.append(data)
        mapping.append([str(counter)]*len(data))
    return data_lists, mapping


def remove_markdown(text):
    """
    Markdown is removed to allow a better prediction from the comet model (not trained with markdown data).
    """
    orig_text = text

    # Remove Markdown links
    # text = re.sub(r'\[.*?\]\(.*?\)', '', text)

    # tables
    text = re.sub(r'\|', '', text)

    # Remove emphasis (bold, italic, strikethrough)
    text = re.sub(r'[_*~]{1,3}', '', text)

    # Remove headings (#, ##, ###, etc.)
    text = re.sub(r'^#+ ', '', text, flags=re.MULTILINE)

    # Remove blockquotes
    # text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r'---', '', text)
    
    return text.strip()

def is_only_punctuation(s):
    # Check if all characters in the string are in the punctuation set
    return all(char in string.punctuation for char in s)

def shorter_than_window(segmented_src, segmented_hyp, segmented_ref):
	window_src = list_to_string(segmented_src) # all of the sentences sent together for comet
	window_hyp = list_to_string(segmented_hyp)
	window_ref = list_to_string(segmented_ref)
	data = [{"src": window_src, "mt": window_hyp, "ref": window_ref}]
	return data

def list_to_string(liste):
	string = ' '.join(liste)
	string = re.sub(r'\n +', '\n', string)
	string = re.sub(r'\n+', '\n', string)
	string = re.sub(r'  +', '\n', string)
	return string

def shorten_if_needed(liste, shortest):
	if len(liste) > shortest:
		# print('text longer than the shortest one')
		dif = len(liste) - shortest
		liste = [' '.join(liste[:dif+1])] + liste[dif+1:]
		return liste
	else: return liste

def bigger_than_window(segmented_src, segmented_hyp, segmented_ref, window_size=3, stride=1):
	window_sources, window_translations, window_references = [], [], []
	for i in range(0, len(segmented_src)- (window_size-1), stride): # stride=1		
		window_src = segmented_src[i:i+window_size] # window_size=3
		window_hyp = segmented_hyp[i:i+window_size]
		window_ref = segmented_ref[i:i+window_size]

		window_sources.append(list_to_string(window_src))
		window_translations.append(list_to_string(window_hyp))
		window_references.append(list_to_string(window_ref))

	data = [{"src": s, "mt": m, "ref":r} for s, m, r in zip(window_sources, window_translations, window_references)]
	return data