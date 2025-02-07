# Cài đặt Python và công cụ cần thiết

## Yêu cầu
- **Python**: Tải từ [python.org](https://www.python.org/downloads/).
- **Code Editor**: Sử dụng VS Code, PyCharm hoặc Jupyter Notebook.

## Cài đặt môi trường ảo (virtual environment)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- Run Application

```bash
python -m uvicorn app.main:app --reload
```
# Technologies Used

### FastAPI
- **Description**: FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.
- **Usage in Project**: It is used to create the API endpoints for the recommendation system, handle HTTP requests, and manage routing.

### Uvicorn
- **Description**: Uvicorn is a lightning-fast ASGI server implementation, using `uvloop` and `httptools`.
- **Usage in Project**: It is used to serve the FastAPI application.

### SQLAlchemy
- **Description**: SQLAlchemy is the Python SQL toolkit and Object-Relational Mapping (ORM) library that gives application developers the full power and flexibility of SQL.
- **Usage in Project**: It is used to manage database connections, define database models, and perform database operations.

### Pydantic
- **Description**: Pydantic is a data validation and settings management library using Python type annotations.
- **Usage in Project**: It is used to validate and parse data coming into the API endpoints and to manage application settings.

### NumPy
- **Description**: NumPy is the fundamental package for array computing with Python.
- **Usage in Project**: It is used for numerical operations and handling arrays in the recommendation algorithms.

### Pandas
- **Description**: Pandas is a fast, powerful, flexible, and easy-to-use open-source data analysis and data manipulation library built on top of the Python programming language.
- **Usage in Project**: It is used to handle and manipulate data fetched from the database for the recommendation algorithms.

### scikit-learn
- **Description**: scikit-learn is a machine learning library for the Python programming language. It features various classification, regression, and clustering algorithms.
- **Usage in Project**: It is used to implement the content-based filtering algorithm for generating recommendations.

### psycopg2
- **Description**: psycopg2 is a PostgreSQL adapter for the Python programming language.
- **Usage in Project**: It is used to connect and interact with the PostgreSQL database.

# Project Structure for Recommendation System

This project uses FastAPI to build a scalable Recommendation System. Below is the folder structure and an explanation of each component.

## Folder Structure

```plaintext
recommendation_system/
├── app/                              # Main application folder
│   ├── __init__.py                   # Initialize Python package
│   ├── api/                          # API route definitions
│   │   ├── __init__.py
│   │   ├── recommendations.py        # Routes for recommendation-related endpoints
│   │   └── dependencies.py           # Dependency management (e.g., database session)
│   ├── core/                         # Application configuration
│   │   ├── __init__.py
│   │   ├── config.py                 # Read environment variables
│   │   └── settings.py               # Define global settings
│   ├── db/                           # Database connection and models
│   │   ├── __init__.py
│   │   ├── session.py                # Manage database connections
│   │   ├── models/                   # Database table models
│   │   │   ├── __init__.py
│   │   │   ├── account.py            # Account table model
│   │   │   ├── course.py             # Course table model
│   │   │   ├── feedback.py           # Feedback table model
│   │   │   └── subscription.py       # Subscription table model
│   ├── schemas/                      # Data validation schemas
│   │   ├── __init__.py
│   │   ├── recommendation.py         # Request/response schema for recommendations
│   │   ├── user.py                   # Schema for user-related data
│   │   └── course.py                 # Schema for course-related data
│   ├── recommendation/               # Core logic for recommendations
│   │   ├── __init__.py
│   │   ├── algorithms.py             # Recommendation algorithms (e.g., Collaborative Filtering)
│   │   ├── services.py               # High-level business logic for recommendations
│   │   └── utils.py                  # Utility functions (e.g., similarity calculations)
│   └── main.py                       # Application entry point
├── .env                              # Environment variables
├── requirements.txt                  # Python dependencies
├── README.md                         # Project documentation
└── run.sh                            # Script to run the application
# bksharing-recommendation
