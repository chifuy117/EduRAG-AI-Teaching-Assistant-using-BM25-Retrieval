# Sample Syllabus Configuration
# This file defines the course structure and topics

SYLLABUS = {
    "DBMS": {
        "topics": [
            "Introduction to Databases",
            "ER Model",
            "Relational Model",
            "Normalization",
            "SQL Queries",
            "Indexing and Hashing",
            "Transactions",
            "Concurrency Control",
            "Recovery",
            "NoSQL Databases"
        ],
        "difficulty_distribution": {
            "easy": ["Introduction to Databases", "ER Model"],
            "medium": ["Relational Model", "SQL Queries", "Transactions"],
            "hard": ["Indexing and Hashing", "Concurrency Control", "Recovery"]
        }
    },
    "OS": {
        "topics": [
            "Introduction to OS",
            "Process Management",
            "CPU Scheduling",
            "Process Synchronization",
            "Deadlocks",
            "Memory Management",
            "Virtual Memory",
            "File Systems",
            "I/O Management",
            "Security"
        ],
        "difficulty_distribution": {
            "easy": ["Introduction to OS", "Process Management"],
            "medium": ["CPU Scheduling", "Memory Management", "File Systems"],
            "hard": ["Process Synchronization", "Deadlocks", "Virtual Memory"]
        }
    },
    "DataStructures": {
        "topics": [
            "Introduction to DS",
            "Arrays and Strings",
            "Linked Lists",
            "Stacks and Queues",
            "Trees",
            "Graphs",
            "Sorting Algorithms",
            "Searching Algorithms",
            "Hash Tables",
            "Dynamic Programming"
        ],
        "difficulty_distribution": {
            "easy": ["Introduction to DS", "Arrays and Strings"],
            "medium": ["Linked Lists", "Stacks and Queues", "Trees"],
            "hard": ["Graphs", "Dynamic Programming", "Hash Tables"]
        }
    },
    "MachineLearning": {
        "topics": [
            "Introduction to Machine Learning",
            "Supervised Learning",
            "Unsupervised Learning",
            "Model Evaluation",
            "Regression",
            "Classification",
            "Clustering",
            "Neural Networks",
            "Feature Engineering",
            "Model Deployment"
        ],
        "difficulty_distribution": {
            "easy": ["Introduction to Machine Learning", "Supervised Learning", "Regression"],
            "medium": ["Classification", "Clustering", "Model Evaluation"],
            "hard": ["Neural Networks", "Feature Engineering", "Model Deployment"]
        }
    }
}

# Subject mapping for consistency
SUBJECT_MAPPING = {
    "database": "DBMS",
    "dbms": "DBMS",
    "databases": "DBMS",
    "operating system": "OS",
    "os": "OS",
    "data structures": "DataStructures",
    "ds": "DataStructures",
    "algorithms": "DataStructures",
    "machine learning": "MachineLearning",
    "ml": "MachineLearning"
}

def get_subject_from_query(query):
    """Extract subject from user query"""
    query_lower = query.lower()

    for keyword, subject in SUBJECT_MAPPING.items():
        if keyword in query_lower:
            return subject

    return None

def validate_topic_in_syllabus(subject, topic):
    """Check if topic exists in subject syllabus"""
    if subject not in SYLLABUS:
        return False

    return topic.lower() in [t.lower() for t in SYLLABUS[subject]["topics"]]

def get_difficulty_for_topic(subject, topic):
    """Get difficulty level for a specific topic"""
    if subject not in SYLLABUS:
        return "medium"

    for difficulty, topics in SYLLABUS[subject]["difficulty_distribution"].items():
        if topic.lower() in [t.lower() for t in topics]:
            return difficulty

    return "medium"