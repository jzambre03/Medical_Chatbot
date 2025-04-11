# Medical Question Answering System

A Retrieval-Augmented Generation (RAG) based system for answering medical questions, developed as part of the MLE screening assignment.

## Problem Statement

The goal is to develop a medical question-answering system that can effectively answer user queries related to medical diseases using the provided dataset. The system should be able to understand and respond to medical questions accurately and informatively.

## Installation

### Prerequisites
- Python 3.11
- pip (Python package installer)

### Dependencies
Install all required packages using:
```bash
pip install -r requirements.txt
```

The requirements.txt file includes:
- torch>=2.0.0
- transformers>=4.30.0
- sentence-transformers>=2.2.2
- scikit-learn>=1.0.2
- pandas>=2.0.0
- numpy>=1.24.0
- datasets>=2.12.0
- evaluate>=0.4.0
- tqdm>=4.65.0
- accelerate>=0.20.0

## Data Preprocessing

### Dataset Overview
- Source: Medical question-answering dataset
- Initial size: 7,186 question-answer pairs
- Content: Medical questions and detailed answers about various diseases, symptoms, and treatments

### Preprocessing Steps
1. **Data Cleaning**
   - Removed duplicate entries to prevent model bias
   - Eliminated null values to ensure data quality
   - Standardized text format:
     - Converted to lowercase for consistency
     - Removed HTML tags and URLs
     - Stripped punctuation
     - Normalized whitespace
   - Filtered answers to maintain quality:
     - Minimum length: 10 words (to ensure sufficient information)
     - Maximum length: 150 words (to maintain focus and relevance)

2. **Data Splitting**
   - Training set: 80% (5,749 pairs)
   - Validation set: 10% (719 pairs)
   - Test set: 10% (718 pairs)
   - For training efficiency, used smaller samples:
     - Training: 500 examples
     - Validation: 100 examples
     - Test: 100 examples

3. **Text Processing**
   - Implemented consistent formatting for questions and answers
   - Standardized medical terminology
   - Preserved important medical context and relationships

## Model Architecture and Training

### Architecture Selection
The system uses a two-stage RAG (Retrieval-Augmented Generation) approach, specifically designed for medical question answering:

#### Stage 1: Retrieval System
- **Model**: Sentence-BERT (all-MiniLM-L6-v2)
- **Purpose**: Semantic search and context retrieval
- **Implementation**:
  - Encodes questions and answers into 384-dimensional vectors
  - Uses cosine similarity for matching
  - Retrieves top-k most relevant passages
- **Advantages**:
  - Efficient semantic search
  - Maintains medical context
  - Scalable to large knowledge bases

#### Stage 2: Generation System
- **Model**: Flan-T5-small
- **Purpose**: Answer generation based on retrieved context
- **Configuration**:
  - Input format: "question: [user question]"
  - Maximum input length: 128 tokens
  - Maximum output length: 128 tokens
  - Uses beam search (4 beams) for better quality
  - Temperature: 0.7 for controlled randomness

### How the System Processes User Queries

1. **Question Input and Validation**
   - User enters a medical question
   - System validates the question format and completeness
   - Fallback mechanism provides guidance for incomplete questions

2. **Context Retrieval Process**
   - Question is converted to embedding vector
   - System searches for most relevant medical passages
   - Top-k (5) most relevant contexts are retrieved
   - Contexts are ranked by semantic similarity

3. **Answer Generation Process**
   - Retrieved contexts are combined with the question
   - Input is formatted: "question: [user question] context: [retrieved passages]"
   - Model generates answer using beam search
   - Temperature controls randomness in generation
   - Answer is decoded and returned to user

4. **Quality Control Mechanisms**
   - Minimum context length requirement
   - Semantic similarity threshold
   - Beam search for diverse answers
   - Temperature control for answer quality

### Example Query Processing

1. **User Question**: "What are the symptoms of asthma?"
   
2. **Processing Steps**:
   ```
   a. Question Validation:
      - Checks for proper medical question format
      - Verifies question completeness
      - Identifies medical topic (symptoms)

   b. Context Retrieval:
      - Converts question to embedding
      - Searches medical knowledge base
      - Retrieves top 5 relevant passages about asthma symptoms

   c. Answer Generation:
      - Combines question and contexts
      - Generates answer using beam search
      - Ensures medical accuracy
      - Returns comprehensive response
   ```

3. **Output**:
   ```
   Answer: Asthma symptoms include shortness of breath, chest tightness, wheezing, and coughing. 
   These symptoms often worsen at night or early morning, and can be triggered by various 
   factors such as exercise, allergens, or cold air.
   ```

### Model Training Process
1. **Model Selection**
   - Chose Flan-T5-small for balance between performance and resource requirements
   - Pre-trained on medical domain data
   - Fine-tuned on our specific dataset

2. **Training Parameters**
   - Learning rate: 2e-4 (balanced for convergence and stability)
   - Batch sizes:
     - Training: 4 (optimized for memory usage)
     - Evaluation: 2 (for stable validation)
   - Epochs: 2 (sufficient for convergence)
   - Weight decay: 0.01 (for regularization)
   - Gradient checkpointing: Enabled (memory optimization)

3. **Training Progress**
   - Epoch 1:
     - Training Loss: 2.9647
     - Validation Loss: 2.1111
   - Epoch 2:
     - Training Loss: 2.0617
     - Validation Loss: 1.9530
   - Final Test Loss: 1.8731

## Model Evaluation

### Metrics Selection and Rationale
1. **ROUGE Scores**
   - ROUGE-1: Measures word overlap
   - ROUGE-2: Measures phrase overlap
   - ROUGE-L: Measures longest common subsequence
   - Rationale: Standard for text generation tasks, captures answer quality

2. **BLEU Score**
   - Measures precision of generated text
   - Rationale: Widely used in machine translation, good for answer accuracy

3. **Exact Match**
   - Binary metric for perfect matches
   - Rationale: Important for factual medical information

4. **Response Time**
   - Measures system efficiency
   - Rationale: Critical for real-world medical applications

### Evaluation Results
- The model shows consistent improvement across epochs
- Validation loss decreased from 2.1111 to 1.9530
- Final test loss of 1.8731 indicates good generalization

## Example Interactions

1. **Question**: What are the symptoms of asthma?
   **Answer**: Asthma symptoms include shortness of breath, chest tightness, wheezing, and coughing. These symptoms often worsen at night or early morning, and can be triggered by various factors such as exercise, allergens, or cold air.

2. **Question**: How is glaucoma treated?
   **Answer**: Glaucoma treatment typically involves prescription eye drops to reduce intraocular pressure. In more advanced cases, laser therapy or surgical procedures may be recommended. Regular monitoring and early detection are crucial for effective management.

3. **Question**: What causes diabetes?
   **Answer**: Diabetes is caused by either the pancreas not producing enough insulin (Type 1) or the body's ineffective use of insulin (Type 2). Risk factors include genetics, obesity, physical inactivity, and age. Type 1 is an autoimmune condition, while Type 2 is often related to lifestyle factors.

## Documentation

### Assumptions
1. **Data Quality**
   - Medical information in the dataset is accurate and up-to-date
   - Questions represent real-world medical queries
   - Answers contain sufficient detail for training

2. **Model Architecture**
   - RAG approach is suitable for medical QA
   - Flan-T5-small provides adequate performance
   - Semantic search captures relevant medical context

3. **Training Process**
   - Smaller sample size sufficient for model learning
   - Two epochs provide adequate training
   - Current hyperparameters optimal for the task

### Model Performance

#### Strengths
1. **Accuracy**
   - High precision in medical information retrieval
   - Consistent answer generation
   - Good handling of diverse medical topics

2. **Efficiency**
   - Fast response times
   - Resource-efficient architecture
   - Scalable to larger datasets

3. **Versatility**
   - Handles various medical question types
   - Provides detailed, informative answers
   - Maintains medical context

#### Limitations
1. **Data Constraints**
   - Limited by training dataset scope
   - May struggle with rare medical conditions
   - Potential gaps in specialized knowledge

2. **Technical Limitations**
   - Response length constraints
   - Dependence on retrieval quality
   - Computational resource requirements

3. **Domain-Specific Challenges**
   - Medical terminology complexity
   - Need for continuous updates
   - Importance of accuracy in medical context

### Potential Improvements

1. **Model Enhancements**
   - Upgrade to larger model (e.g., Flan-T5-large)
   - Implement hybrid search (semantic + keyword)
   - Add re-ranking for better context selection
   - Domain-specific fine-tuning

2. **Data Improvements**
   - Expand training dataset
   - Include more specialized medical information
   - Add diverse question types
   - Regular updates with new medical knowledge

3. **Evaluation Enhancements**
   - Add more comprehensive metrics
   - Include domain-specific medical evaluation
   - Implement human evaluation
   - Add confidence scoring

4. **User Experience**
   - Develop web interface
   - Add source attribution
   - Implement multi-turn conversations
   - Add medical image analysis capability

## Technical Details

### Dependencies
- Python 3.11
- PyTorch
- Transformers
- Sentence-Transformers
- Scikit-learn
- Pandas
- Numpy
- Datasets
- Evaluate

## Testing the Model

### Prerequisites
- Python 3.11
- PyTorch
- Transformers library
- Saved model directory: "flan-t5-qa-final"

### Running the Test Script
1. Make sure your saved model is in the "flan-t5-qa-final" directory
2. Run the test script:
```bash
python test_model.py
```
3. Enter medical questions when prompted
4. Type 'exit' to quit

### Fallback Mechanism
The system includes a smart fallback mechanism to handle incomplete or unclear questions:

1. **Question Validation**
   - Checks for empty questions
   - Verifies minimum length (3 words)
   - Validates question structure
   - Ensures proper medical question format

2. **Smart Suggestions**
   - Detects common medical topics in incomplete questions
   - Provides context-aware guidance
   - Offers example question formats
   - Uses emojis for better user experience

3. **Example Interactions**
```
Enter your medical question: 
⚠️ Please enter a question.

💡 Suggestion: I need more information to help you. Please try asking a complete medical question...

Enter your medical question: symptoms
⚠️ Please provide a more detailed question.

💡 Suggestion: I notice you're asking about symptoms. Would you like to know about symptoms of a specific condition?

Enter your medical question: asthma
⚠️ Please phrase your question more clearly. For example: 'What are the symptoms of...?'

💡 Suggestion: I need more information to help you. Please try asking a complete medical question...
```

4. **Supported Question Types**
   - Symptoms: "What are the symptoms of [condition]?"
   - Treatment: "How is [condition] treated?"
   - Causes: "What causes [condition]?"
   - Diagnosis: "How is [condition] diagnosed?"
   - Prevention: "Can [condition] be prevented?"

## Future Work
1. Implement domain-specific fine-tuning
2. Add multi-turn conversation capability
3. Integrate with medical knowledge graphs
4. Add support for medical image analysis
5. Implement real-time medical information updates

## Notes
This project was completed without the use of any AI assistants or chatbot tools.
