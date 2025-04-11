from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re

def load_model(model_path):
    """Load the saved model and tokenizer"""
    print("Loading model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    return tokenizer, model

def is_valid_question(question):
    """Check if the question is valid and complete"""
    # Remove extra whitespace
    question = question.strip()
    
    # Check if question is empty
    if not question:
        return False, "Please enter a question."
    
    # Check if question is too short (less than 3 words)
    if len(question.split()) < 3:
        return False, "Please provide a more detailed question."
    
    # Check if question has proper structure (contains a question word or ends with ?)
    question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can', 'does', 'is', 'are']
    if not any(word in question.lower().split() for word in question_words) and not question.endswith('?'):
        return False, "Please phrase your question more clearly. For example: 'What are the symptoms of...?'"
    
    return True, ""

def generate_answer(question, tokenizer, model, max_length=128):
    """Generate answer for a given medical question"""
    # Format input
    input_text = f"question: {question}"
    
    # Tokenize input
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding="max_length"
    )
    
    # Generate answer
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            temperature=0.7,
            do_sample=True
        )
    
    # Decode and return answer
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

def get_fallback_response(question):
    """Provide helpful suggestions for incomplete questions"""
    question = question.strip().lower()
    
    # Check for common medical topics
    medical_topics = {
        'symptoms': 'Would you like to know about symptoms of a specific condition?',
        'treatment': 'Would you like to know about treatment options for a specific condition?',
        'cause': 'Would you like to know about causes of a specific condition?',
        'diagnosis': 'Would you like to know about diagnosis methods for a specific condition?',
        'prevention': 'Would you like to know about prevention methods for a specific condition?'
    }
    
    # Check if question contains any medical topics
    for topic, response in medical_topics.items():
        if topic in question:
            return f"I notice you're asking about {topic}. {response}"
    
    # Default fallback response
    return """I need more information to help you. Please try asking a complete medical question, such as:
- What are the symptoms of [condition]?
- How is [condition] treated?
- What causes [condition]?
- How is [condition] diagnosed?
- Can [condition] be prevented?"""

def main():
    # Load model
    model_path = "flan-t5-qa-final"
    tokenizer, model = load_model(model_path)
    
    print("\nMedical QA System - Testing Mode")
    print("Type 'exit' to quit")
    print("-" * 50)
    
    while True:
        # Get user question
        question = input("\nEnter your medical question: ").strip()
        
        if question.lower() == 'exit':
            break
            
        # Validate question
        is_valid, error_message = is_valid_question(question)
        
        if not is_valid:
            print(f"\n⚠️ {error_message}")
            print("\n💡 Suggestion:", get_fallback_response(question))
            continue
            
        # Generate and display answer
        try:
            answer = generate_answer(question, tokenizer, model)
            print("\nAnswer:", answer)
        except Exception as e:
            print(f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    main() 