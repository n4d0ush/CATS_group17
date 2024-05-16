#!/usr/bin/env python3
"""Reproduce your result by your saved model.

This is a script that helps reproduce your prediction results using your saved
model. This script is unfinished and you need to fill in to make this script
work. If you are using R, please use the R script template instead.

The script needs to work by typing the following commandline (file names can be
different):

python3 run_model.py -i unlabelled_sample.txt -m model.pkl -o output.txt

"""

# author: Chao (Cico) Zhang
# date: 31 Mar 2017

import argparse
import pickle
import sys

import joblib
# Start your coding

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# End your coding


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Reproduce the prediction')
    parser.add_argument('-i', '--input', required=True, dest='input_file',
                        metavar='unlabelled_sample.txt', type=str,
                        help='Path of the input file')
    parser.add_argument('-m', '--model', required=True, dest='model_file',
                        metavar='model.pkl', type=str,
                        help='Path of the model file')
    parser.add_argument('-o', '--output', required=True,
                        dest='output_file', metavar='output.txt', type=str,
                        help='Path of the output file')
    # Parse options
    args = parser.parse_args()

    if args.input_file is None:
        sys.exit('Input is missing!')

    if args.model_file is None:
        sys.exit('Model file is missing!')

    if args.output_file is None:
        sys.exit('Output is not designated!')

    # Start your coding

    # Load validation
    validation_data = pd.read_table(args.input_file)  # Shows the data with samples as rows, so need to transpose
    validation_data = validation_data.T  # Transposes data, so that samples are now rows.
    validation_data = validation_data.iloc[4:, :]  # This is the complete cleaned up dataset

    # Restrict data to selected features
    feats = [12, 15, 17, 32, 308, 354, 474, 480, 484, 485, 487, 489, 499, 554, 555, 558, 594, 610, 612, 615, 623, 674, 679, 693, 700, 718, 724, 725, 727, 729, 733, 743, 758, 849, 874, 998, 1001, 1105, 1281, 1295, 1383, 1423, 1598, 1606, 1635, 1636, 1638, 1641, 1645, 1650, 1651, 1655, 1660, 1679, 1681, 1682, 1684, 1690, 1701, 1862, 1877, 1879, 1896, 1902, 1910, 1913, 1950, 1963, 1973, 2047, 2056, 2154, 2168, 2183, 2184, 2185, 2196, 2205, 2213, 2220, 2221, 2223, 2226, 2241, 2281, 2285, 2293, 2379, 2661, 2663, 2709, 2722, 2733, 2751, 2760, 2763, 2765, 2774, 2831, 2833]
    validation_data = validation_data.iloc[:, feats]

    # Load random forest classifier and perform classifications on validation data
    rf_classifier = joblib.load(args.model_file)

    predictions = rf_classifier.predict(validation_data)
    # Write predictions into file
    i = 0
    with open(args.output_file, 'w') as file:
        file.write('"Sample"' + '\t' + '"Subgroup"' + '\n')
        for prediction in predictions:
            if prediction == 0:
                file.write(f'"{validation_data.index[i]}"' + '\t' + '"HER2+"' + '\n')
            elif prediction == 1:
                file.write(f'"{validation_data.index[i]}"' + "\t" + '"HR+"' + '\n')
            elif prediction == 2:
                file.write(f'"{validation_data.index[i]}"' + "\t" + '"Triple Neg"' + '\n')
            i += 1
    # End your coding

if __name__ == '__main__':
    main()

