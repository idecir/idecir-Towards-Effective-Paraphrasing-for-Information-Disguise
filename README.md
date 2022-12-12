# Towards Effective Paraphrasing for Information Disguise
This repository contains the code for our **ECIR 2023 accepted paper: `Towards Effective Paraphrasing for Information Disguise`.**

### Repository structure:
* `code/beam_search_code/Disguise Text.ipynb` : Shows the disguise of a true sentence (query) via our model
* `code/beam_search_code/beam_helper`: contains all the helper modules for our model
    * `beam_utils.py`: contains the code dealing with single level phrase substitution, Beam Search, Constituency Parse Tree creation etc.
    * `synonyms_store.py`: contains the code to get synonyms of a term in Counterfitting synonyms vector space
    * `faiss_fetch.py`: Contains the code for initializing DPR and fetching top K relevant documents
    * `perplexity_calculation.py`: contains the code initiating the perplexity calculation
    * `fetch_use_scores.py`: contains the code to create Universal Sentence Encoding for a given piece of text
* `code/beam_search_code/counter-fitted-vectors.txt`: Counterfitting vectors used for fetching synonyms
* `data/all_syns.json`: Contains the 10 nearest neighbours for all terms in the dictionary (the nearest neighbours were calcuated by using `Facebook AI Similarity Search (FAISS)`) on the vectors in `counter-fitted-vectors.txt`
* `sql_lite_dbs/<name>.db`: expects the database containing the metadata and contents of the document store (to be used by DPR)
* `code/faiss_indexes/<name>.faiss`: expects the vectors for the documents in the document store
* `code/faiss_indexes/exp_with_two_thou_short.json`: expects the configuration file containing the parameters describing how to read "<name>.faiss" 

### Requirements
Details of the `conda environment` for the above codebase is present in `adversarial_search.yaml`. 
We use Haystack's DPR implementation.

### Attack parameters which can be modified/passed to `Class BeamSearch` in `beam_utils.py`    



<table>
  <tr>
   <td><strong>Parameter Name</strong>
   </td>
   <td><strong>Description</strong>
   </td>
  </tr>
  <tr>
   <td>MAX_DEPTH
   </td>
   <td>Number of levels in the beam search tree ie the MAXIMUM number of phrase substitutions allowed to be made in the query
   </td>
  </tr>
  <tr>
   <td>ALPHA_VAL
   </td>
   <td>
<ul>

<li>Weighing parameter (to weight semantic similarity to the original query and locatibility differently). 

<li>It is used in the calculation of score for a node in the BeamSearchTree. 

<li>See the section `Algorithm Explanations` in the paper for the details.
</li>
</ul>
   </td>
  </tr>
  <tr>
   <td>NUM_PERPLEXITY_NODES_TO_EXPAND
   </td>
   <td>
<ul>

<li> Number of nodes in the Constituency Parse Tree to be considered for attacking.

<li>Corresponds to the parameter "P"  in STEP 3 of Section 3.1  of the paper.
</li>
</ul>
   </td>
  </tr>
  <tr>
   <td>BeamWidth
   </td>
   <td>Max number of nodes at each level of the beam tree.
   </td>
  </tr>
  <tr>
   <td>NUM_FAISS_DOCS_TO_RETRIEVE
   </td>
   <td>Max relevant documents to be fetched for the query in which the source document's presence needs to be checked.
   </td>
  </tr>
  <tr>
   <td>SIMILARITY_CUT_OFF_THRESHOLD
   </td>
   <td>
<ul>

<li>Candidates which have a similarity of less than  `SIMILARITY_CUT_OFF_THRESHOLD` with the original sentence are filtered out.

<li>Corresponds to `epsilon` in the paper
</li>
</ul>
   </td>
  </tr>
</table>
