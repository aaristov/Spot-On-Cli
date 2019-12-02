import os
from fastspt import matimport as mt
import argparse
import json
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def pprint(d: dict):
    for k, i in d.items():
        print(f'{k} : {i}')


def main(args=None, callback=None):

    def log(msg=None, level='info'):
        if callback:
            callback({'Message': msg})
            print(msg)
        logger.__getattribute__(level)(msg)

    def debug(msg):
        return log(msg, level='debug')

    def error(msg):
        return log(msg, level='error')

    def info(msg):
        return log(msg, level='info')

    # Main parser

    parser = argparse.ArgumentParser(
        prog='fastspt',
        description='Fit kinetics using Spot-ON method')
    subparsers = parser.add_subparsers(dest='command')

    for command in ['fit']:

        subparsers.add_parser(command)
    # fit

    fit_parser = subparsers.add_parser(
        'fit',
        help='fit kinetics model using spot-on')
    fit_parser.add_argument('--mat', nargs='*', type=str, default='',
                            help='matlab dataset with localizations, \
                                format replicate->fov->coodrinates \
                                [[[[x, y, frame, track]]]], \
                                units [px, px, int, int]')
    fit_parser.add_argument('--xml', nargs='*', type=str, default='',
                            help='xml tracks from Trackmate')

    fit_parser.add_argument('--config', type=str, help='json config file')

    # args = parser.parse_args()
    debug(args)
    try:
        if args is not None:
            args = parser.parse_args(args)
        else:
            args = parser.parse_args()

    except TypeError:
        error(f'Wrong args while parsing: {args}')

        exit(1)

    if not args.command:
        error('No command, exiting')
        # parser.print_help()

    elif args.command == 'fit':
        info('Do fit')
        fnames = args.matfile

        try:
            params = json.load(open(args.config))
            info('Current configuration:')
            pprint(params)

        except json.JSONDecodeError:
            error('Bad configuration')
            exit(2)

        for fname in fnames:
            if not os.path.isfile(fname):
                error('File not found')
                exit(1)
            elif not isinstance(params, dict):
                error(f'bad config {params}')
                exit(2)
            else:
                print('Start analysis')
        try:
            mt.analyse_mat_files(fnames, **params)
            exit(0)
        except KeyboardInterrupt:
            info('User interrupt, exiting')
            exit(1)
    else:
        pass


if __name__ == "__main__":
    main()
