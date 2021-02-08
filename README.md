# Detection of Iterative Adversarial Attacks via Counter Attack

Deep neural networks (DNNs) have proven to be powerful tools for processing unstructured data. However for high-dimensional data, like images, they are inherently vulnerable to adversarial attacks. Small almost invisible perturbations added to the input can be used to fool DNNs. Various attacks, hardening methods and detection methods have been introduced in recent years. Notoriously, Carlini-Wagner (CW) type attacks computed by iterative minimization belong to those that are most difficult to detect. 
We provide a mathematical proof that the CW attack can be used as a detector itself. That is, under certain assumptions and in the limit of attack iterations this detector provides asymptotically optimal separation of original and attacked images. In numerical experiments, we experimentally validate this statement. 

For further reading, please refer to https://arxiv.org/abs/2009.11397. 

# Run Code:
```python3
python3 run_attacks.py
```

Before running the code, please edit all necessary paths stored in "global_defs.py" and install CleverHans. For installation details, we refer to https://github.com/cleverhans-lab/cleverhans.


# Author:
Kira Maag (University of Wuppertal)



