import json
from typing import List, Dict


class synonyms_index:
    def __init__(self, readymade_dict_path: str):
        ##################################
        with open(readymade_dict_path, 'r') as fd:
            # this file has the 10 nearest neighbours for all the words in counterfitting vectors dictionary
            # the neighbours were generated using the Counterfitting embeddings
            # present at https://github.com/nmrksic/counter-fitting
            self.syn_dict: Dict = json.load(fd)
            # print("Number of keys in premade file is ", len(self.syn_dict))

    def fetch_k_nearest(self, q_word: str, k_val: int) -> List[str]:
        '''Expects `q_word` to be in lower case.'''

        # if q_word not in self.word_dict:
        if q_word not in self.syn_dict:
            print("Word not present in dictionary")
            return []

        # The first nearest neighbour is discarded as the 1st nearest neighbour
        # is equal to the q_word itself
        alt_words: List[str] = self.syn_dict[q_word][:k_val]
        return alt_words


if __name__ == "__main__":
    syn_obj = synonyms_index(
        readymade_dict_path="../../../data/all_syns.json")
    word = "computer"
    ans = syn_obj.fetch_k_nearest(word, 10)
    print(ans)
