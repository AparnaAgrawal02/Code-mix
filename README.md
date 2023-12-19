# Code-mix
## Abstract
 Text generation is a highly active area of research in the computational linguistic
 community.Code-mixing, the phenomenon of mixing two or more languages in a single
 utterance, is a common occurrence in multilingual communities worldwide. With the
 growing interest in developing natural language processing systems that can handle
 code-mixed text, recent research has focused on generating code-mixed text using neu-
 ral models. However, generating high-quality code-mixed text is a challenging task due
 to the complex nature of language mixing.
 In this study, we explore the use of curriculum training to fine-tune multilingual
 Transformers like mT5 and mBert. We also use a dependency-free strategy for pro-
 ducing code-mixed texts from bilingual distributed representations that we utilize for
 enhancing language model performance due to the dearth of training data for code-
 mixing. We apply a curriculum learning strategy, in particular, where we first train
 the language models on synthetic data, then on gold code-mixed data as suggested by
 the paper https://arxiv.org/pdf/2105.08807.pdf
## Keywords: 
 CodeMix, multilingual, fine-tuning, synthetic ,Real,Transformers

## Qualitative Analysis
 ### Example of nearest neighbours:  
 – aur, and: similarity=0.9570194482803345  
 – of, ki: similarity=0.7434549927711487   
 ### Eample of synthetic sentence:  
 • en: ”Okay just calm down, we’ll get to the bottom of this.”,  
 cm: ”Okay just shAMta ho ham ’ll get to the bottom of this.  

##  Sample
 ### Input’s Text  
    sister will eat green mango today   
 ### codemix (True Value)  
    Didi aaj hare rang ke aam khaengi  
 ### mt5 on PHNC dataset  
    bhai green mango peene ke saath kharab  
 ### mt5 on synthetic data  
    agararikaghose bhU.Nge will eat green mAta Aja the hindu   
 ### mt5(curriculum training)  
    behen aaj hare mango pee jaate hai the hindu  
 we can clearly see the improvement in curriculum training


 ## How to run?
  python mt5_inference.py --model "checkpoint" --text  

