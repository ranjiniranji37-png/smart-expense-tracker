# Smart Expense Tracker (AI-Powered)

An improved version of my Module 2 Expense Tracker, built as part of the **Codomax Digital Solutions AI & ML Internship - Module 5 (AI Tools & Mini Project)**.

The original tracker asked the user to type a category for every expense. In this version, a **Machine Learning model predicts the category automatically** from the expense title.

## How the AI Works
1. `training_data.csv` has 145 labeled expense titles across 7 categories (Food, Travel, Shopping, Bills, Entertainment, Health, Education).
2. Titles are converted to numbers using **TF-IDF** (word and character n-grams).
3. A **Multinomial Naive Bayes** classifier (Scikit-learn) learns from this data.
4. When a new expense is added, the model predicts the category with a confidence score.
5. If confidence is high, the user can accept with Enter. If the AI is unsure, it asks the user to type the category.

## Model Accuracy
5-fold cross-validation accuracy: **about 62%** on this small dataset. Accuracy improves as more labeled data is added, and the user can always correct a wrong prediction.

## Features
- AI category prediction with confidence score
- Add, view and delete expenses
- Category-wise summary with percentages
- Data saved in `expenses.json`
- Input validation and error handling

## Project Structure
| File | Description |
|------|-------------|
| `smart_expense_tracker.py` | Main application |
| `category_model.py` | ML model (train, evaluate, predict) |
| `training_data.csv` | Labeled training data |
| `AI_TOOLS_REPORT.md` | How AI tools helped during the internship |
| `requirements.txt` | Python libraries needed |

## How to Run
```bash
pip install -r requirements.txt
python smart_expense_tracker.py
```

To check the model on its own:
```bash
python category_model.py
```

## Improvements Over Module 2
| Module 2 | Module 5 |
|----------|----------|
| User types category manually | AI predicts category automatically |
| No machine learning | TF-IDF + Naive Bayes model |
| Single file | Separate model and application files |

## Future Improvements
- Add more training data and more categories
- Learn from the user's corrections
- Monthly budget alerts and charts

## Author
Ranjini K
