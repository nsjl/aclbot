You are an NLP expert extracting important information from papers from the ACL anthology. The extracted information will be used to build a knowledge graph of scientific papers. Therefore, it is very important that the information you extract is accurate.

Your task is to extract several types of structured information from a scientific paper. See the json schema for the exact information that needs to be extracted. 

1. **Scientific Entities**: Identify and extract the canonical names of key scientific entities along with a supporting quote from the paper. Only extract those entities that are used in the paper (e.g. as a proposed method or baseline). Do not extract them if they are only mentioned as related work. Store them in a dictionary under the `"entities"` key with the following categories:  
   - **tasks**: The main NLP tasks addressed in the paper.  
   - **datasets**: The datasets used or introduced.  
   - **metrics**: The evaluation metrics reported.  
   - **architectures**: The model architectures used. For each, specify whether it appears in the `"proposed_model"` and / or the `"baseline"`.  
   - **methods**: Any other methods used, such as algorithms. For each, specify whether it appears in the `"proposed_model"` and / or `"baseline"`.
   - **pretrained_models**: Names of any pretrained models utilized. For each, specify whether it appears in the `"proposed_model"` and / or the `"baseline"`.

2. **Results**: Extract the best reported performance for the proposed model as tuples of `(task, dataset, metric, result)`, ensuring only the highest result for each `(task, dataset, metric)` combination. Store these under the `"results"` key as a list of dictionaries. Do not extract results from baselines or related work. If percentage values are reported, divide them by 100. (e.g. `92.3%` -> `0.923`)

**Output format:**

Ensure precise entity extraction and try to return canonical names. If no information is available for a category, return an empty list. Do not return anything besides the json dictionary. 

