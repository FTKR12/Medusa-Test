import os
from prettytable import PrettyTable

from utils.options import get_args
from utils.seed import set_seed
from utils.logger import setup_logger
from src import build_runner

def main(args, logger):
    runner = build_runner(args)
    output = runner.run()

    table = PrettyTable(field_names=output.keys())
    table.add_row(list(output.values()))
    logger.info(f'\n{table}')

if __name__ == '__main__':
    args = get_args()
    args.output_dir = f"{args.output_dir}/{args.name}"
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    logger = setup_logger('Mdeusa Test', save_dir=args.output_dir)
    logger.info(str(args).replace(',','\n'))

    main(args, logger)