You are an NLP expert extracting important information from papers from the ACL anthology. The extracted information will be used to build a knowledge graph of scientific papers. Therefore, it is very important that the information you extract is accurate.

Extract the following structured information from the given research paper:  

**Scientific Entities**: Identify and extract the canonical names of key scientific entities along with a supporting quote from the paper. Only extract those entities that are used in the paper (e.g. as a proposed method or baseline). Do not extract them if they are only mentioned as related work.  Store them in a dictionary under the `"entities"` key with the following categories:  
   - **tasks**: The main NLP tasks addressed in the paper.  
   - **datasets**: The datasets used or introduced.  
   - **metrics**: The evaluation metrics reported.  
   - **architectures**: The model architectures used. For each, specify whether it appears in the `"proposed_model"` and / or a `"baseline"`.
   - **methods**: Any other methods used, such as algorithms. For each, specify whether it appears in the `"proposed_model"` and / or a `"baseline"`.
   - **pretrained_models**: Names of any pretrained models utilized. For each, specify whether it appears in the `"proposed_model"` and / or a `"baseline"`.

**Output format:**  

{
  "entities": {
    "tasks": [{"entity": "Named Entity Recognition", "quote": "We evaluate our approach on the Named Entity Recognition task."}],
    "datasets": [{"entity": "CoNLL-2003", "quote": "We use the CoNLL-2003 dataset for training and evaluation."}],
    "metrics": [{"entity": "F1-score", "quote": "Our model achieves an F1-score of 92.3%."}],
    "architectures": [{"entity": "Transformer", "quote": "Our method is based on a Transformer architecture.", "usage": ["proposed_model"]}],
    "methods": [{"entity": "Expectation-Maximization", "quote": "As done by Smith et al., we optimize the parameters of their model using Expectation-Maximization.", "usage": ["baseline"]}],
    "pretrained_models": [{"entity": "BERT", "quote": "We fine-tune BERT for this task and compare to the non-fine-tuned model.", "usage": ["proposed_model", "baseline"]}]
  }
}

Ensure precise entity extraction and try to return canonical names. If no information is available for a category, return an empty list. Only extract an entity or a result if you are absolutely sure. Do not return anything besides the json dictionary. 

