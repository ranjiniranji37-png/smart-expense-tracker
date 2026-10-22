"""
Smart Expense Tracker (AI-Powered)
Codomax Digital Solutions AI & ML Internship - Module 5
Author: Ranjini K

An improved version of the Module 2 Expense Tracker. It uses a Machine Learning
model to automatically predict the category of each expense.
"""

import json
import os
from datetime import date

from category_model import train_model, predict_category

DATA_FILE = "expenses.json"
CONFIDENCE_THRESHOLD = 0.45


def load_expenses():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return []


def save_expenses(expenses):
    with open(DATA_FILE, "w") as file:
        json.dump(expenses, file, indent=4)


def add_expense(expenses, model):
    title = input("Enter expense title: ").strip()
    if not title:
        print("Title cannot be empty.")
        return
    try:
        amount = float(input("Enter amount: "))
        if amount <= 0:
            print("Amount must be greater than 0.")
            return
    except ValueError:
        print("Invalid amount. Please enter a number.")
        return

    category, confidence = predict_category(model, title)
    print(f"AI suggests category: {category} ({confidence * 100:.0f}% confident)")

    if confidence >= CONFIDENCE_THRESHOLD:
        choice = input("Press Enter to accept, or type a different category: ").strip()
    else:
        choice = input("AI is not sure. Please enter the category: ").strip()
    if choice:
        category = choice.title()

    expenses.append({
        "title": title,
        "category": category,
        "amount": amount,
        "date": str(date.today()),
    })
    save_expenses(expenses)
    print(f"Saved under '{category}'.")


def view_expenses(expenses):
    if not expenses:
        print("No expenses recorded yet.")
        return
    print("\n{:<4} {:<12} {:<20} {:<14} {:>10}".format("No", "Date", "Title", "Category", "Amount"))
    print("-" * 64)
    for i, e in enumerate(expenses, start=1):
        print("{:<4} {:<12} {:<20} {:<14} {:>10.2f}".format(
            i, e["date"], e["title"][:19], e["category"], e["amount"]))
    print("-" * 64)


def delete_expense(expenses):
    view_expenses(expenses)
    if not expenses:
        return
    try:
        number = int(input("Enter expense number to delete: "))
        removed = expenses.pop(number - 1)
        save_expenses(expenses)
        print(f"Deleted: {removed['title']}")
    except (ValueError, IndexError):
        print("Invalid expense number.")


def show_summary(expenses):
    if not expenses:
        print("No expenses recorded yet.")
        return
    totals = {}
    for e in expenses:
        totals[e["category"]] = totals.get(e["category"], 0) + e["amount"]
    total = sum(totals.values())
    print("\n--- Expense Summary ---")
    for category, amount in sorted(totals.items(), key=lambda x: x[1], reverse=True):
        print(f"{category:<14} {amount:>10.2f}  ({amount / total * 100:.1f}%)")
    print("-" * 32)
    print(f"{'Total':<14} {total:>10.2f}")
    print(f"Highest spending category: {max(totals, key=totals.get)}")


def main():
    print("Loading AI category model...")
    model = train_model()
    expenses = load_expenses()

    while True:
        print("\n===== SMART EXPENSE TRACKER (AI) =====")
        print("1. Add Expense (AI predicts category)")
        print("2. View Expenses")
        print("3. Delete Expense")
        print("4. Summary by Category")
        print("5. Exit")
        choice = input("Choose an option (1-5): ").strip()

        if choice == "1":
            add_expense(expenses, model)
        elif choice == "2":
            view_expenses(expenses)
        elif choice == "3":
            delete_expense(expenses)
        elif choice == "4":
            show_summary(expenses)
        elif choice == "5":
            print("Thank you for using Smart Expense Tracker!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
