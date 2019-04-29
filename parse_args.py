import argparse

def parse_args():
    # create the top-level parser
    parser = argparse.ArgumentParser(prog='PCAM')

    subparsers = parser.add_subparsers(help='sub-command help')

    parser_train = subparsers.add_parser('train', help='a help')
    parser_train.add_argument('--model-name', '-m', dest='model_name',
                              required=True, help='Model name (for loading custom model or for transfer learning)')
    parser_train.add_argument('--training-kind', '-k', dest='training_kind',
                              choices=['new', 'resume', 'resume_causal'],
                              help="'new' to train new model; 'resume' to resume previous training; 'resume_causal' to resume only the CNN part of the model (drops the stroke data). The options 'resume' and 'resume_causal' require also the 'run-id' argument (see 'run-id')" )
    parser_train.add_argument('--run-id', dest='run_id',
                              help='Specify which model to use when resuming training')
    parser_train.add_argument('--num-epochs', '-E', dest='num_epochs',
                              default=100, help='Model name (for loading custom model or for transfer learning)')
    parser_train.add_argument('--max-stale', '-S', dest='max_stale',
                              default=10, type=int,
                              help='Early stopping: number of epochs without improvement before stopping')
    parser_train.add_argument('--local', action='store_true', help='if running locally')
    parser_train.add_argument('--test-run', action='store_true',
                              help='if making a test run (uses smaller dataset)')
    parser_train.add_argument('--negative-only', action='store_true',
                              dest='negative_only',
                              help='only use negative data (for debugging purposes')
    parser_train.set_defaults(kind="train")

    parser_eval = subparsers.add_parser('evaluate', help='b help')
    parser_eval.add_argument('--local', action='store_true')
    parser_eval.set_defaults(kind="eval")

    return vars(parser.parse_args())
