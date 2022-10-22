from beam_helper.ext_bert_fun import (
    get_word_probabilities,
    sum_log_probabilities)


def fetch_pll_scores(bert_tokenizer, bert_model, txt: str) -> dict:
    '''Returns the pseudo log likelihood of the sentence'''
    res = get_word_probabilities(
        txt,
        bert_tokenizer=bert_tokenizer,
        bert_model=bert_model)
    obj = dict()
    obj['org_scores'] = res
    obj['final_score'] = sum_log_probabilities(res)
    return obj
