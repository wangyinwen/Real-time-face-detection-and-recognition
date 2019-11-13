import os
import random
import logging
import time
import sys
#import util.utils as utils
import argparse


def main(args):
    diff_size = 3000
    combin_size = 200
    all_list = []
    nodir_list = []

    face_logtime = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))
    logging_config('../logs/pairs/create_pairs_' + face_logtime + '.log')

    if (os.path.exists(args.pairs_path)):
        os.remove(args.pairs_path)
        logging.info("delete the specified file......")

    with open(args.pairs_path, 'w') as file:
        # create all combination list
        file.write('10	300' + '\n')
        for x in os.listdir(args.root_dir):
            child_dir = os.path.join(args.root_dir, x)
            person_list = []
            if os.path.isdir(child_dir):
                person_list.append(x)
                for i in os.listdir(child_dir):
                    #person_list.append(i.split('a')[1].split('_')[0])
                    person_list.append(i)
                all_list.append(person_list)
            else:
                nodir_list.append(x)

        logging.info("all validation data list :" + str(all_list))
        logging.info("no dictroy file list :" + str(nodir_list))

        # create same person lines
        all_same_lines = []
        for x, item in enumerate(all_list):
            if len(item) < 3:
                continue
            for i in range(1, len(item)):
                for j in range(i + 1, len(item)):
                    line = item[0] + ' ' + item[i] + ' ' + item[j]
                    all_same_lines.append(line)
        same_lines = random.sample(all_same_lines, 3000)
        logging.info("all same lines combines:" + str(len(all_same_lines)))

        # create different persons lines
        combine_list = []
        random_sample = random.sample(all_list, combin_size)
        list_len = len(random_sample)
        for i in range(0, list_len):
            len1 = len(random_sample[i])
            for j in range(i + 1, list_len):
                len2 = len(random_sample[j])
                for m in range(1, len(random_sample[i])):
                    for n in range(1, len(random_sample[j])):
                        combine_list.append(
                            random_sample[i][0] + ' ' + random_sample[i][m] + ' ' + random_sample[j][0] + ' ' +
                            random_sample[j][n])

        logging.info("combine_list_len:" + str(len(combine_list)))
        diff_lines = random.sample(combine_list, diff_size)

        # generate pairs by the  LFW's rule (300 matched pairs and 300 mismatched pairs within each set,10sets)
        a, b, handle = 0, 0, 0
        while a + b < 6000:
            if (handle == 0):
                for i in range(a, a + 300):
                    file.write(same_lines[i] + '\n')
                    handle = 1
                a += 300
            elif (handle == 1):
                for i in range(b, b + 300):
                    file.write(diff_lines[i] + '\n')
                    handle = 0
                b += 300


def logging_config(logfile):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", type=str,
                        help='Path to the data directory containing aligned your face patches.',
                        default='/home/hyz/testface')
    parser.add_argument('--pairs_path', type=str,
                        help='Path to save pairs.txt', default='../data/my_newpairs.txt')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
