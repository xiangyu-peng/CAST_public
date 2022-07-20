### Fine-tune LM
#### Finetune GPT-2 on roc
- We use the new dataset: `cometFilter/data_story/100KStories_dealed.txt`
```ruby
python cometFilter/finetune_language_model/finetune_gpt2_roc.py
```
- Saved checkpoints are in `cometFilter/finetune_language_model/finetuned_model/roc_pure`
- epoch = 2

#### Fine-tune GPT-2-roc on generation with prompt
- prompt processed files: `cometFilter/data_story/prompt_deal.txt`
```ruby
python cometFilter/finetune_language_model/finetune_on_roc.py
```
- Saved checkpoints are in `cometFilter/finetune_language_model/finetuned_model/roc_char`
- epoch = 3

#### Fine-tune GPT-2-wp on generation without prompt
File: `data_story/writingPrompts/all.txt`
```ruby
python cometFilter/finetune_language_model/finetune_on_roc.py
```
- Saved checkpoints are in `cometFilter/finetune_language_model/finetuned_model/wp_pure`
- epoch = 3

#### Finetune with RL and use comet as classifier
```ruby
python generate_continuation.py
```