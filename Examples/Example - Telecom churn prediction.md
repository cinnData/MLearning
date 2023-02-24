# Example - Telecom churn prediction

## Introduction 

The term **churn** is used in marketing to refer to a customer leaving the company in favor of a competitor. Churning is a common concern of **Customer Relationship Management** (CRM). A key step in proactive churn management is to predict whether a customer is likely to churn, since an early detection of the potential churners helps to plan a **retention campaign**.

The objective of this example is to develop a churn model for a telecom company. The data set is based on a random sample of 7,032 customers. 1,869 of those customers churned during the last month, a **churning rate** of 26.6%. The data includes information about: 

* Customers who left within the last month.

* Services that each customer has signed up for: multiple lines, online security, etc. 

* Customer account information:  how long they've been a customer, contract, payment method, etc.

* Demographic info about customers: gender, age range, etc.

## The data set

The data come in the file `telecom.csv`. The variables included in the data set are: 

* `id`, customer's unique ID.

* `gender`, customer's gender (Female/Male).

* `senior_citizen`, a dummy for being more than 60 years of age.

* `partner`, whether the customer has a partner (Yes/No).

* `dependents`, whether the customer has dependents (Yes/No).

* `tenure`, number of months the customer has stayed with the company.

* `phone_service`, whether the customer has a phone service (Yes/No).

* `multiple_lines`, whether the customer has multiple lines (Yes/No/No phone service).

* `internet_service`, customer's internet service provider (DSL/Fiber optic/No).

* `online_security`, whether the customer is signed up for online security (Yes/No/No internet service).

* `online_backup`, whether the customer is signed up for online backup (Yes/No/No internet service).

* `device_protection`, whether the customer is signed up for device protection (Yes/No/No internet service).

* `tech_support`, whether the customer is signed up for technical support (Yes/No/No internet service).

* `streaming_tv`, whether the customer is signed up for streaming TV (Yes/No/No internet service).

* `streaming_movies`, whether the customer is signed up for streaming movies (Yes/No/No internet service).

* `contract`, type of contract (Month-to-month/One year/Two year).

* `paperless_billing`, whether the customer is signed up for paper billing (Yes/No).

* `payment_method`, customer's payment method (Bank transfer/Credit card/Electronic\break check/Mailed check).

* `monthly_charges`, amount charged to the customer last month, in US dollars.

* `total_charges`, total amount charged to the customer, in US dollars.

* `churn`, whether the customer has churned (Yes/No).

Source: Kaggle.

## Questions

Q1. Develop a logistic regression model to calculate a churn score for each customer.

Q2. How is the distribution of churn scores? Is it different for the churners and the non-churners?

Q3. Set an adequate cutoff for the churn scores and apply it to decide which customers are potential churners. What is the true positive rate? And the false positive rate?

Q4. Validate your model using a train/test split.
