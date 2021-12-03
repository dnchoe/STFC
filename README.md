# STFC (Ship to First Commit)
Code for cleansing data for Ship to First Commit metric to be consumed by Tableau. The precise STFC definition is described later at this page.

## More detailed description of the metrc and the code:
This code is a part of Ship-To-First-Commit (STFC) metric. For the metric, the first promise date needs to be recognized for each sales line. (The first promise dates are the dates a company first promised to its customers when it will deliver the products.) As the data table has all the records if there is any change in lines, i.e. there could be multiple rows regarding to one sales number, we need data cleansing code to get the line with the first promise date.

## Precise definition
<a href="https://www.codecogs.com/eqnedit.php?latex=STFC=\frac{STFC\_Hit}{STFC\_Hit&plus;STFC\_Miss}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?STFC=\frac{STFC\_Hit}{STFC\_Hit&plus;STFC\_Miss}" title="STFC=\frac{STFC\_Hit}{STFC\_Hit+STFC\_Miss}" /></a>

where
- STFC Hit: Shipped within $\pm$ 7 days of 1st Promise Date (inclusive)
- STFC Miss:
  - Shipped: ship date outside
  - Backlog: Current date > 1st PD + 7 days
- 1st PD is the first commit or 1st PD after a customer changes CRD
  - Split orders keeps the original 1st PD of its parents
- Report date = 1st PD + 7 days

## What needs to be improved for STFC
This code is not the final version. The final version cannot be released becasue it has been done as a part of a company project. For the accurate metric reporting, the following logic needs to be applied to the code: while tracing back to each sales lines, if there is any customer requester date (CRD) change, then the first promise date tracing needs be stopped. It is because the reasonable first commit is the first promise date after the CRD change. The current code, however, doesn't stop tracing back even if there is a CRD change.

## Other applications of the code
This code could be further developed to calculate the number of schedule shipment date changes, i.e. SSD churns, and customer request date changes, i.e. CRD churns, considering the order line splits.
