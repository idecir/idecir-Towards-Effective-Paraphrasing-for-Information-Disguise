# Towards Effective Paraphrasing for Information Disguise
This repository contains the code for our **ECIR 2023 submission: `Towards Effective Paraphrasing for Information Disguise`.**
See the **attack results of our approach** on some queries from our dataset in [Examples](#eg_ref)

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
    
    
# Examples showcasing the performance of our Information Disguise Approach
<a name="eg_ref"></a>

## **Example 1:**


<table>
  <tr>
   <td colspan="5" ><strong>Original Sentence</strong>
   </td>
  </tr>
  <tr>
   <td colspan="5" >I bought a tablet about a month ago and since then I have returned it to the store 2 times and gonna return it a third time now and ask for another swap.
   </td>
  </tr>
  <tr>
   <td colspan="5" ><strong>Attack summary</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Beam Level ID</strong>
   </td>
   <td><strong>Generated sentence with the best attack</strong>
   </td>
   <td><strong>Attack sequence</strong>
   </td>
   <td><strong>Similarity with original sentence</strong>
   </td>
   <td><strong>Document rank on generated sentence</strong>
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>I bought a tablet about a month ago and since then I have returned it to the store 2 times and gonna return it a third time now and ask for another <strong>exchanging</strong> .
   </td>
   <td>
<ul>

<li>swap -> exchanging
</li>
</ul>
   </td>
   <td>0.905
   </td>
   <td>8
   </td>
  </tr>
  <tr>
   <td>2
   </td>
   <td>I bought a tablet about a month ago and since then I have returned it to the <strong>boutique</strong> 2 times and gonna return it a third time now and ask for <strong>payment</strong> .
   </td>
   <td>
<ul>

<li>another swap -> payment

<li>store -> boutique
</li>
</ul>
   </td>
   <td>0.802
   </td>
   <td>18
   </td>
  </tr>
  <tr>
   <td>3
   </td>
   <td>I bought a tablet about a month ago and since then I have returned it to the <strong>boutique</strong> 2 times and <strong>would</strong> return it a third time now and ask for another <strong>exchanging</strong> .
   </td>
   <td>
<ul>

<li>swap -> exchanging

<li>gonna -> would

<li>store -> boutique
</li>
</ul>
   </td>
   <td>0.929
   </td>
   <td>Outside top 20
   </td>
  </tr>
</table>



---


## **Example 2:**


<table>
  <tr>
   <td colspan="5" ><strong>Original Sentence</strong>
   </td>
  </tr>
  <tr>
   <td colspan="5" >There are two large file cabinets in the office for just that reason.
   </td>
  </tr>
  <tr>
   <td colspan="5" ><strong>Attack summary</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Beam Level ID</strong>
   </td>
   <td><strong>Generated sentence with the best attack</strong>
   </td>
   <td><strong>Attack sequence</strong>
   </td>
   <td><strong>Similarity with original sentence</strong>
   </td>
   <td><strong>Document rank on generated sentence</strong>
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>There are two large <strong>speaker</strong> cabinets in the office for just that reason .
   </td>
   <td>
<ul>

<li>file->speaker
</li>
</ul>
   </td>
   <td>0.807
   </td>
   <td>2
   </td>
  </tr>
  <tr>
   <td>2
   </td>
   <td>There are two large <strong>speaker</strong> cabinets in the <strong>bureau</strong> for just that reason .
   </td>
   <td>
<ul>

<li>file->speaker

<li>office -> bureau
</li>
</ul>
   </td>
   <td>0.805
   </td>
   <td>14
   </td>
  </tr>
  <tr>
   <td>3
   </td>
   <td>There <strong>been</strong> two <strong>grands</strong> file cabinets in the <strong>bureau</strong> for just that reason .
   </td>
   <td>
<ul>

<li>are -> been

<li>large -> grands

<li>office -> bureau
</li>
</ul>
   </td>
   <td>0.945
   </td>
   <td>Outside top 20
   </td>
  </tr>
</table>



---


## **Example 3:**


<table>
  <tr>
   <td colspan="5" ><strong>Original Sentence</strong>
   </td>
  </tr>
  <tr>
   <td colspan="5" >It was the first meeting of the families and honestly, it went pretty well - my parents mentioned that they will be invited to a dinner party and we finally settled on a date which is next week.
   </td>
  </tr>
  <tr>
   <td colspan="5" ><strong>Attack summary</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Beam Level ID</strong>
   </td>
   <td><strong>Generated sentence with the best attack</strong>
   </td>
   <td><strong>Attack sequence</strong>
   </td>
   <td><strong>Similarity with original sentence</strong>
   </td>
   <td><strong>Document rank on generated sentence</strong>
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>It was the first meeting of the families and honestly, it went pretty well - my parents mentioned that they will be invited to a dinner party and we finally <strong>settles</strong> on a date which is next week .
   </td>
   <td>
<ul>

<li>settled -> settles
</li>
</ul>
   </td>
   <td>0.99
   </td>
   <td>1
   </td>
  </tr>
  <tr>
   <td>2
   </td>
   <td>It was the first meeting of the <strong>familial</strong> and honestly, it went pretty well - my parents <strong>talked</strong> that they will be invited to a dinner party and we finally settled on a date which is next week .
   </td>
   <td>
<ul>

<li>mentioned -> talked

<li>families -> familial
</li>
</ul>
   </td>
   <td>0.986
   </td>
   <td>5
   </td>
  </tr>
  <tr>
   <td>3
   </td>
   <td>It was the first meeting of the <strong>familial</strong> and honestly, it went pretty well - my parents <strong>talked</strong> that they will be invited to <strong>attend</strong> and we finally settled on a date which is next week 
   </td>
   <td>
<ul>

<li>mentioned -> talked

<li>families -> familial

<li>a dinner party -> attend
</li>
</ul>
   </td>
   <td>0.9724
   </td>
   <td>Outside top 20
   </td>
  </tr>
</table>



---


## **Example 4:**


<table>
  <tr>
   <td colspan="5" ><strong>Original Sentence</strong>
   </td>
  </tr>
  <tr>
   <td colspan="5" >today my mom called me and asked me to fly down from michigan to florida to help her and her fiance unpack their new home.
   </td>
  </tr>
  <tr>
   <td colspan="5" ><strong>Attack summary</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Beam Level ID</strong>
   </td>
   <td><strong>Generated sentence with the best attack</strong>
   </td>
   <td><strong>Attack sequence</strong>
   </td>
   <td><strong>Similarity with original sentence</strong>
   </td>
   <td><strong>Document rank on generated sentence</strong>
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>today my mom called me and asked me to fly down from michigan to florida to help her and her <strong>bridegroom</strong> unpack their new home .
   </td>
   <td>
<ul>

<li>fiance -> bridegroom
</li>
</ul>
   </td>
   <td>0.978
   </td>
   <td>2
   </td>
  </tr>
  <tr>
   <td>2
   </td>
   <td>today my mom called me and asked me to fly down from <strong>hawaii</strong> to florida to help her and her <strong>bridegroom</strong> unpack their new home .
   </td>
   <td>
<ul>

<li>fiance -> bridegroom

<li>michigan -> hawaii
</li>
</ul>
   </td>
   <td>0.927
   </td>
   <td>18
   </td>
  </tr>
  <tr>
   <td>3
   </td>
   <td>today my mom called me and asked me to fly down from <strong>ohio</strong> to florida to help her and her <strong>bridegroom</strong> <strong>at</strong> their new home .
   </td>
   <td>
<ul>

<li>fiance -> bridegroom

<li>unpack -> at

<li>michigan -> ohio
</li>
</ul>
   </td>
   <td>0.943
   </td>
   <td>Outside top 20
   </td>
  </tr>
</table>



---


## **Example 5:**


<table>
  <tr>
   <td colspan="5" ><strong>Original Sentence</strong>
   </td>
  </tr>
  <tr>
   <td colspan="5" >She started getting heated and I said idc how you want to act or dress up in private but when you are wearing a tail in public or a furry suit in public you deserve the ridicule and bullying you deserve.
   </td>
  </tr>
  <tr>
   <td colspan="5" ><strong>Attack summary</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Beam Level ID</strong>
   </td>
   <td><strong>Generated sentence with the best attack</strong>
   </td>
   <td><strong>Attack sequence</strong>
   </td>
   <td><strong>Similarity with original sentence</strong>
   </td>
   <td><strong>Document rank on generated sentence</strong>
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>She started getting heated and I said idc how you want to act or dress up in private but when you are wearing a tail in public or a <strong>black</strong> suit in public you deserve the ridicule and bullying you deserve .
   </td>
   <td>
<ul>

<li>furry -> black
</li>
</ul>
   </td>
   <td>
<ul>

<li>0.968
</li>
</ul>
   </td>
   <td>3
   </td>
  </tr>
  <tr>
   <td>2
   </td>
   <td>She started getting heated and I said idc how you want to act or dress up in private but when you are wearing a <strong>gown</strong> in public or a <strong>matching</strong> suit in public you deserve the ridicule and bullying you deserve 
   </td>
   <td>
<ul>

<li>furry -> black

<li>tail -> gown
</li>
</ul>
   </td>
   <td>
<ul>

<li>0.907
</li>
</ul>
   </td>
   <td>20
   </td>
  </tr>
  <tr>
   <td>3
   </td>
   <td>She started getting heated and I said idc how you want to act or dress up in private but when you are wearing <strong>suits</strong> or a <strong>fancy</strong> suit in public you deserve the <strong>mockery</strong> and bullying you deserve .
   </td>
   <td>
<ul>

<li>furry -> fancy

<li>tail in public -> suits

<li>ridicule -> mockery
</li>
</ul>
   </td>
   <td>
<ul>

<li>0.906
</li>
</ul>
   </td>
   <td>Outside top 20
   </td>
  </tr>
</table>



---


## **Example 6:**


<table>
  <tr>
   <td colspan="5" ><strong>Original Sentence</strong>
   </td>
  </tr>
  <tr>
   <td colspan="5" >The backstory: My grandma has always been fiercely independent, lives alone (but close to everyone, about 3 miles away) and retired about 6 months ago.
   </td>
  </tr>
  <tr>
   <td colspan="5" ><strong>Attack summary</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Beam Level ID</strong>
   </td>
   <td><strong>Generated sentence with the best attack</strong>
   </td>
   <td><strong>Attack sequence</strong>
   </td>
   <td><strong>Similarity with original sentence</strong>
   </td>
   <td><strong>Document rank on generated sentence</strong>
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>The backstory : My grandma has always been fiercely independent , lives alone ( but close to <strong>everybody</strong> , about 3 miles away ) and retired about 6 months ago .
   </td>
   <td>
<ul>

<li>everyone -> everybody
</li>
</ul>
   </td>
   <td>0.997
   </td>
   <td>1
   </td>
  </tr>
  <tr>
   <td>2
   </td>
   <td>The backstory : My <strong>grandmom</strong> has always been fiercely independent , lives alone ( but close to <strong>each</strong> , about 3 miles away ) and retired about 6 months ago .
   </td>
   <td>
<ul>

<li>everyone -> everybody

<li>grandma -> grandmom
</li>
</ul>
   </td>
   <td>0.976
   </td>
   <td>11
   </td>
  </tr>
  <tr>
   <td>3
   </td>
   <td>The backstory : My <strong>grandmom</strong> has always been fiercely independent , lives alone ( but close to <strong>each</strong> , about 3 miles away ) and <strong>retire</strong> about 6 months ago .
   </td>
   <td>
<ul>

<li>everyone -> each

<li>grandma -> grandmom

<li>retired -> retire
</li>
</ul>
   </td>
   <td>0.972
   </td>
   <td>Outside top 20
   </td>
  </tr>
</table>



---


## **Example 7:**


<table>
  <tr>
   <td colspan="5" ><strong>Original Sentence</strong>
   </td>
  </tr>
  <tr>
   <td colspan="5" >Sorry If I make any mistakes Here's some brief context: My mom was never good with money.
   </td>
  </tr>
  <tr>
   <td colspan="5" ><strong>Attack summary</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Beam Level ID</strong>
   </td>
   <td><strong>Generated sentence with the best attack</strong>
   </td>
   <td><strong>Attack sequence</strong>
   </td>
   <td><strong>Similarity with original sentence</strong>
   </td>
   <td><strong>Document rank on generated sentence</strong>
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>Sorry If I make any mistakes Here 's some brief <strong>backgrounds</strong> : My mom was never good with money .
   </td>
   <td>
<ul>

<li>context -> backgrounds
</li>
</ul>
   </td>
   <td>0.9109
   </td>
   <td>6
   </td>
  </tr>
  <tr>
   <td>2
   </td>
   <td>Sorry If I make any mistakes Here 's some brief <strong>marco</strong> : My <strong>moms</strong> was never good with money .
   </td>
   <td>
<ul>

<li>context -> macros

<li>mom -> moms
</li>
</ul>
   </td>
   <td>0.8117
   </td>
   <td>19
   </td>
  </tr>
  <tr>
   <td>3
   </td>
   <td>Sorry If <strong>je</strong> make any mistakes Here 's some <strong>nutshell</strong> <strong>backgrounds</strong> : My mom was never good with money .
   </td>
   <td>
<ul>

<li>context -> backgrounds

<li>brief -> nutshell

<li>I -> je
</li>
</ul>
   </td>
   <td>0.9103
   </td>
   <td>Outside top 20
   </td>
  </tr>
</table>



---


## **Example 8:**


<table>
  <tr>
   <td colspan="5" ><strong>Original Sentence</strong>
   </td>
  </tr>
  <tr>
   <td colspan="5" >We get updates for new policies and programs we're doing fairly often and we're now calling customers and asking if they want to sign up for the latest program.
   </td>
  </tr>
  <tr>
   <td colspan="5" ><strong>Attack summary</strong>
   </td>
  </tr>
  <tr>
   <td><strong>Beam Level ID</strong>
   </td>
   <td><strong>Generated sentence with the best attack</strong>
   </td>
   <td><strong>Attack sequence</strong>
   </td>
   <td><strong>Similarity with original sentence</strong>
   </td>
   <td><strong>Document rank on generated sentence</strong>
   </td>
  </tr>
  <tr>
   <td>1
   </td>
   <td>We get <strong>online</strong> and we're now calling customers and asking if they want to sign up for the latest program .
   </td>
   <td>
<ul>

<li>updates for new policies and programs we're doing fairly often -> online
</li>
</ul>
   </td>
   <td>0.808
   </td>
   <td>3
   </td>
  </tr>
  <tr>
   <td>2
   </td>
   <td>We get updates for new policies and programs we're doing fairly often and we <strong>get</strong> now calling customers and asking if they want to sign up for the <strong>lately</strong> program .
   </td>
   <td>
<ul>

<li>‘re -> get

<li>latest -> lately
</li>
</ul>
   </td>
   <td>0.979
   </td>
   <td>7
   </td>
  </tr>
  <tr>
   <td>3
   </td>
   <td>We get updates for new policies and programs we <strong>were</strong> doing <strong>too</strong> often and we 're now calling customers and asking if they want to sign up for the <strong>lately</strong> program .
   </td>
   <td>
<ul>

<li>‘re -> get

<li>latest -> lately

<li>fairly -> too
</li>
</ul>
   </td>
   <td>0.938
   </td>
   <td>Outside top 20
   </td>
  </tr>
</table>



---

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
