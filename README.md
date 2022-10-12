## CAST
This code accompanies the paper [Inferring the Reader: Guiding Automated Story Generation with Commonsense Reasoning](https://arxiv.org/abs/2105.01311)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

<img src="https://github.com/anonymous-hero/CAST/blob/master/images/figure_1.png" alt="drawing" width="400"/>

## COMeT Filtering on GPT-2
![alt text](https://github.com/anonymous-hero/CAST/blob/master/images/pipeline.png)
### :building_construction: Installation
- There are some extra packages you need to install here.

```ruby    
conda install -c pytorch pytorch
pip install ftfy
```

- First, download the pretrained models from [here](https://drive.google.com/open?id=1FccEsYPUHnjzmX-Y5vjCBeyRt1pLo8FB).

- Then untar the file:

```ruby 
tar -xvzf pretrained_models.tar.gz
```

- Then run the following script to interactively generate arbitrary ATOMIC event effects:
```ruby 
python scripts/interactive/atomic_single_example.py --model_file pretrained_models/atomic_pretrained_model.pickle
```

#### How to use [COMeT2020](https://github.com/allenai/comet-atomic-2020)?

| Relations | Human Readable Template |
| :--- | :--- |
| AtLocation | located or found at/in/on |
| CapableOf | is/are capable of |
| Causes | causes |
| CausesDesire | makes someone want |
| CreatedBy | is created by |
| Desires | desires |
| HasA | has, possesses or contains |
| HasFirstSubevent | BEGINS with the event/action |
| HasLastSubevent | ENDS with the event/action |
| HasPrerequisite | to do this, one requires |
| HasProperty | can be characterized by being/having |
| HasSubEvent | includes the event/action |
| HinderedBy | can be hindered by |
| InstanceOf | is an example/instance of |
| isAfter | happens after |
| isBefore | happens before |
| isFilledBy | blank can be filled by |
| MadeOf | is made of |
| MadeUpOf | made (up) of |
| MotivatedByGoal | is a step towards accomplishing the goal |
| NotDesires | do(es) NOT desire |
| ObjectUse, UsedFor | used for |
| oEffect | as a result, Y or others will |
| oReact | as a result, Y or others feels |
| oWant | as a result, Y or others want |
| PartOf | is a part of |
| ReceivesAction | can receive or be affected by the action |
| xAttr | X is seen as |
| xEffect | as a result, PersonX will |
| xIntent | because PersonX wanted |
| xNeed | but before, PersonX needed |
| xReact | as a result, PersonX feels |
| xReason | because |
| xWant | as a result, PersonX wants |

#### How to use [GPT-2](https://github.com/huggingface/transformers)?

- Run the code to use gpt-2
```ruby
python ./examples/text-generation/run_generation.py --model_type=gpt2 --length=20 --model_name_or_path=gpt2 --num_return_sequences 5
```
#### How to use them together to do COMeT filtering with GPT-2?

- All the related file is in /cometFilter/ folder.

- Remember to add path in `comet-commonsense/src/interactive/functions.py` in your machine

- Run the baseline of COMeT filtering: No filtering, two characters' interaction story

- Run codes with new decoder, new prompt system, single char. 
Add `-t` to use two chars. 
Remove '-d' to remove decoder.
  - `--model_name_or_path`: str; the language model path.
  - `--prompt_file`: str; the first story sentence file. Each line is one prompt. txt file.
  - `--comet_use`: store_true; whether to use our technique. Remove it to use baseline.
  - `-f`: str; filter level. We use `strong_new` in inlg paper.
  - `--coref_use`: store_true;  use coreference resoluation to remove the 3rd char
  - `--use_cond`: store_true;  use new prompt style. EX. ` * [Char_1] * [Char_1] loves ice cream.`
  - `-l`: int; story length. We use 5 or 10.
  - `-d`: store_true; new decoding system.
  - `--history_use`: store_true; use history when generate story using LM.
  - `--diverse`: 
  

```ruby
python -W ignore cometFilter.py --model_name_or_path finetune_language_model/finetuned_model/roc_char/21 --prompt_file data_story/test.txt --comet_use -f strong_new --use_cond -l 5 -d --history_use --diverse
```
   
- Run codes with new decoder, `old prompt system`. Add `-t` to use two chars.
```ruby
python -W ignore cometFilter.py --model_name_or_path finetune_language_model/finetuned_model/roc_pure --prompt_file data_story/test.txt --comet_use -f strong_new -l 5 -d --history_use --diverse
```

- Run baseline to compare with.
```ruby
python -W ignore cometFilter.py --model_name_or_path finetune_language_model/finetuned_model/roc_char/21 --prompt_file data_story/test.txt --use_cond -l 5 --history_use --device_id 1
```
### :woman_technologist: Matching Criteria

- Decide matching criteria pairs by the following code:

```ruby
 python comet_2020_use.py --check
```

`--check`: print all the matching criteria pairs' max score

- Decide criteria matching criteria with files.

```ruby
 python comet_2020_use.py --file_check
```

`--file_check`: summarize all the possible matching criteria pairs in one file.

`--story_file_path`: file path of story you use for decide matching criteria.

`--save_file_path`: file path of story you use for saving matching criteria.

- Verify matching criteria by the following code:

```ruby
 python comet_2020_use.py --verify --char 2
```

`--verify`: verify the matching criteria pairs

`--char`: int; number of characters

#### Single Character :woman: :man:

| Prompt | Continuation |
| :--- | :--- |
| xWant | xIntent |
| _sentence_ | xNeed|
| xEffect| _sentence_|
| CausesDesire| Desires|
| isBefore| isAfter|
| AtLocation| AtLocation|
    
#### Multiple Character :couple: :two_men_holding_hands: :two_women_holding_hands:

| Prompt | Continuation |
| :--- | :--- |
| oReact | xAttr |
| oWant | xIntent|
| oEffect | _sentence_ |
    
### :tipping_hand_woman: Features
  
- **Filter level**:
    - `--comet_use`: T or F.
    - `--filter_level` : str.
        - `weak`: As long as any 1st sentence's "effects on others" matches the 2nd sentence's "effects/ attr/causes for PersonX", return True
        - `medium`: As long as any 1st sentence's "effects on others" matches the 2nd sentence's "attr/causes for PersonX", return True
        - `strong`: oWant -> xIntent; oEffect -> xNeed; oReact -> xAttr
        - `want`: oWant -> xIntent
        - `need`: oEffect -> xNeed
        - `effect`: oReact -> xAttr
    - Relax the match condition to weak (`--random_tech`) when there is no match after 50 rounds
    - `--num_matching`: int, default is 1, indicating how many matching has to be reached.
    
- **Parser** for extracting character names (`spacy==2.3.2` and `neuralcoref==4.0`)
        
        pip install neuralcoref       
        python -m spacy download en
               
    - spacy.strings.StringStore size changed error: **try installing from source**
    
- **[Similarity check](https://github.com/UKPLab/sentence-transformers)** for matching

        pip install sentence_transformers
    
    - The similarity file is located at `CommonsenseStoryGen/cometFilter/similarity.py`
    - It embeds the output from the COMET and calculate the similarity score for matching.
    - The default similarity score threshold is 0.8 (`--sim_threshold`).

- **[Diverse beam search](http://vision.soic.indiana.edu/papers/diversebeam2018aaai.pdf)**

    - download this weight to the `/beam_search` folder

            curl -O  "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
 
    - Run the codes with diverse beam search (hamming)
        
        - `--beam_search`: diverse beam search type, i.e. hamming (#TODO for more options)
        - `--beam_lamb`: the coefficient for adding penalty, larger is more penalty
    
    - Modify the words prob
        - do NOT decrease the prob of high frequency words (TFIDF) - added
        - do NOT decrease the prob of characters 
            - `--exclude_chars`: bool, to control if exclude chars names
    
    - Use similarity to penalize:
    
        - `--glove_use`: bool, to control whether to use glove embedding
                       
- **Filter parser**: add parser to filter process
  
    - Use the parser to check if the sentence candidate generate the third characters.
    - `--filter_parser`: T or F

- **Coreference resolution**:
    
    - We use Coreference Resolution in spaCy with Neural Networks [here](https://github.com/huggingface/neuralcoref/blob/master/README.md).
    - If we find the third char or like 'them', some coref which is not among the two chars, we filter it out.
    - `--coref_use` : T or F
    - Only choose **one** between **coref** and **filter parser**.
    
- :heavy_heart_exclamation: Use the whole history to generate text instead of one sentence!
    
    - `--history_use`: T or F, default is T
    
- Baseline with no chars name.
    
    - We replace the name of characters with [MALE], [FEMALE], [NEURAL] and train the gpt2, located in `/gpt2_models/roc_model_generalized`
    - --char_name_use: bool, default=True. Use F/ False if you wanna use this model
       
- Number of characters

    - We define this model is used to generate a 2-char or 1-char story generation.
    - `--two_char_use`: bool, default is `True` , if you only wanna consider one character, pls set it as `False`
- Backtrack

    - We use backtrack to help find the matching?
    - `--backtrack_use`: bool. True indicates use the backtrack.


#### Fine-tune LM
- RocStories with ``[CHAR1]`` and ``[CHAR2]``:`cometFilter/data_story/100KStories_dealed.txt`
- prompt dealed files: `cometFilter/data_story/prompt_deal.txt`
- Fine-tune LM on roc dataset:
    ```ruby
    python cometFilter/finetune_language_model/finetune_gpt2_roc.py
    ```
- Fine-tune LM on prompt_char:
    ```ruby
    python cometFilter/finetune_language_model/finetune_on_roc.py
    ```
####
- Run the parser, you need to go to `./stanford-corenlp-4.1.0`
and then run `java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 30000`
