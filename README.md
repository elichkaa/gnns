Differentiable Graph Construction for Latent Graph Learning
-----------------------------------------------------------

### Problem

*   Graph topology is inherently discrete—an edge either exists or it does not—which interrupts gradient flow
    

  

### Task

*   Goal: achieve differentiable graph flow
    

*   Implement a complete end-to-end differentiable pipeline
    
    *   Text Encoder -> Differentiable Graph Module -> GNN -> Classifier
        
    *   Text Encoder: A pre-trained encoder (e.g., DistilBERT) converts raw text into token embeddings
        
    *   Differentiable Graph Learner: Induces a sparse adjacency matrix A from embeddings to construct a graph
        
    *   Graph Neural Network: GNN to encode the derived latent graph (you can choose and test any GNN you like, e.g. GAT)
        
    *   Classifier: Global pooling aggregates node features into a document vector, then MLP predicts class.
        
*   Compare differentiable graph construction mechanisms (Gumbel-Softmax vs. continuous metric learning)
    
*   Evaluate whether learning task-specific latent graphs improves multi-class classification performance
    

  

### Key Components

*   \[1\] DGM —> Kazi et al., "Differentiable Graph Module (DGM) for Graph Convolutional Networks," 2020.
    
*   \[2\] IDGL —> Chen et al., "Iterative Deep Graph Learning for Graph Neural Networks," NeurIPS 2020.
    

  

### Dataset - 20Newsgroups

*   It consists of ~18,800 newsgroup posts across 20 topical categories
    
*   [https://scikit-learn.org/0.19/datasets/twenty\_newsgroups.html](https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html)
    
*   [https://huggingface.co/datasets/SetFit/20\_newsgroups](https://huggingface.co/datasets/SetFit/20_newsgroups)
    

  

### Approach

*   Replicate the results from IDGL (add link)
    
*   Compare DGM (add link) on the same task - cDGM vs. dDGM
    
*   Regularization in these settings is crucial as learned graphs often degenerate (fully connected or disconnected)
    

  

### Ablation Studies

*   Sparsity: k in {5, 10, 20} neighbors per node
    
*   Residual: With vs. without skip connection from encoder
    
*   GNN type: Validate GAT vs GCN performance (expecting GAT > GCN)
    
*   Qualitative Analysis Visualize induced graphs for 3-5 documents.
    
*   Do edges connect topically related words?
    
*   Are there "hub" tokens (keywords)?
    
*   Do graph structures differ between classes?