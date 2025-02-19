
import sys

def dolfin_doc_to_text(doc):
	langs = {
		'en': 'English',
		'de': 'German',
		'fr': 'French',
		'es': 'Spanish',
		'it': 'Italian'
	}
	src_lang = langs[doc['src_lang']]
	trg_lang = langs[doc['trg_lang']]

	return f"Translate from {src_lang} into {trg_lang}. Only provide the translation without any other comment. The following is the text to translate:\n\n{doc['source_text']}"
