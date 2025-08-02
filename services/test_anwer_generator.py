# test_answer_generator.py
from answer_generator import AnswerGenerator

if __name__ == "__main__":
    generator = AnswerGenerator()

    # Example insurance-related question
    question = "How many hospitals are included in the cashless network?"


    # Simulated document chunks from vector DB (provided by Member 3)
    retrieved_chunks = [
        {
            "text": "Bajaj Finserv Health Insurance provides comprehensive coverage for various medical needs. However, maternity expenses are covered only under specific plans like the 'Family Health Plus' plan, and a waiting period of 2 years applies."
        },
        {
            "text": "The standard plan excludes maternity coverage unless explicitly mentioned. Please refer to the policy document or contact customer care for exact inclusions."
        },
        {
            "text": "All policyholders are eligible for cashless treatment at over 6500 network hospitals across India."
        }
    ]

    # Parsed query can be extended later; currently it's not used
    parsed_query = {}

    # Generate and print the answer
    answer = generator.generate_answer(question, parsed_query, retrieved_chunks)
    print("\nGenerated Answer:\n", answer)
