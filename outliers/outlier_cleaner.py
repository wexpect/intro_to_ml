#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    cleaned_data = []

    ### your code goes here

    errors = np.abs(predictions - net_worths)

    idx = errors[:, 0] < np.percentile(errors, 90)

    cleaned_data = [ages[idx], net_worths[idx], errors[idx]]

    return cleaned_data

