import argparse
import csv
import lmdb
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed

parser = argparse.ArgumentParser()
parser.add_argument('lmdb')
parser.add_argument('chunks', type=int)
parser.add_argument('prefix')
args = parser.parse_args()

env = lmdb.open(args.lmdb)
entries = env.stat()['entries']
print(f'There are {entries} entries in the lmdb file')
print(
    f'There will be {args.chunks} chunks with {entries/args.chunks} for each')


def convert_lmdb(env, filename, id, rows, prefix):
    with env.begin() as txn:
        with txn.cursor() as curs, open(filename, 'w') as csvfile:
            writer = None
            count = 0
            for i in rows:
                row = pd.read_msgpack(
                    curs.get(f'{i}'.encode()), encoding='ascii').to_dict()
                if not writer:
                    writer = csv.DictWriter(csvfile, fieldnames=row.keys())
                    writer.writeheader()
                writer.writerow(row)
                count += 1
                if count % 1000 == 0:
                    print(f'Wrote {count} rows to {filename}')
    return filename


chunks = np.array_split(range(entries), args.chunks)

with ThreadPoolExecutor() as executor:
    futures_to_id = {executor.submit(
        convert_lmdb,
        env,
        f'{args.prefix}.{i}.csv',
        i,
        chunks[i],
        args.prefix
    ): i for i in range(len(chunks))}
    for future in as_completed(futures_to_id):
        id = futures_to_id[future]
        try:
            filename = future.result()
        except Exception as e:
            print(f'file {id} generated an exception: {e}')
        else:
            print(f'{filename} done')
