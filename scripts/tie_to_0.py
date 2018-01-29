import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('in_csv')
parser.add_argument('out_csv')
args = parser.parse_args()

with open(args.in_csv, 'r') as in_csv, open(args.out_csv, 'w') as out_csv:
    reader = csv.reader(in_csv)
    writer = csv.writer(out_csv)
    for row in reader:
        if row[-1] == '0.5':
            row[-1] = '0'
        writer.writerow(row)

print('Done')
