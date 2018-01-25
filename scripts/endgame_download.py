import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen


def download(i):
    url = 'http://moodle.usm.md/endgame/getpgn.php?pid=%d' % i
    response = urlopen(url)
    data = response.read()
    return data.decode('latin-1')


parser = argparse.ArgumentParser()
parser.add_argument('max_pid', type=int)
parser.add_argument('out_pgn')
args = parser.parse_args()


with ThreadPoolExecutor() as executor, open(args.out_pgn, 'w') as f:
    futures = [executor.submit(download, i) for i in range(1, args.max_pid)]
    count = 0
    for future in as_completed(futures):
        count += 1
        f.write(future.result())
        f.write('\n')
        if count % 50 == 0:
            print(f'{count} downloaded')
    print('Done')
