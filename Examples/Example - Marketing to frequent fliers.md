# Example - Marketing to frequent fliers

## Introduction 

The file `fliers.csv` contains information on 3,999 passengers who belong to the EastWest's **frequent flier program**. For each passenger, the data include information on their mileage history and on different ways they accrued or spent miles in the last year. The goal is to identify **segments of passengers** that have similar characteristics for the purpose of targeting different segments for different types of mileage offers. Key issues are their flying patterns, earning and use of frequent flier rewards, and use of the airline credit card.

## The data set

The variables included in the data set are: 

* `id`, a unique customer ID.

* `balance`, number of miles eligible for award travel.

* `qual_miles`, number of miles counted as qualifying for Topflight status.

* `cc1_miles`, number of miles earned with frequent flier credit card in the past 12 months, coded as 1 (under 5,000), 2 (5,000-10,000), 3 (10,001-25,000), 4 (25,001-50,000).

* `cc2_miles`, number of miles earned with Rewards credit card in the past 12 months, coded as above.

* `cc3_miles`, number of miles earned with Small Business credit card in the past 12 months, coded as above.

* `bonus_miles`, number of miles earned from non-flight bonus transactions in the past 12 months.

* `bonus_trans`, number of non-flight bonus transactions in the past 12 months.

* `flight_miles_12mo`, number of miles flight miles in the past 12 months.

* `flight_trans_12mo`, number of flight transactions in the past 12 months.

* `days_since_enroll`, number of days since the customer was enrolled.

* `award`, a dummy for having an award.

Source: G Shmueli and PC Bruce (2016), based upon a real business case and real data. The company name has been changed.

## Questions

Q1. Group the customers in four segments using the information available.

Q2. The same, but after normalizing the variables. Can you describe the segments in a few words?
