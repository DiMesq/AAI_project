# Generate concrete data for the one-shot 20-way classification task
# with the evaluation set (using omniglot split)
import sys


def main(path_to_alphabets):
    # get all alphabets
    # get dict of alphabets to num characters in that alphabet
    no_replacement_target_images = set()
    test_episodes = []
    for i in range(5000):
        pass
        # sample alphabet
        # sample target character
        # sample query image (one that hasn't been used before)
        # sample target image (one that hasn't been used before)
        # add query and target images to no_replacement
        # sample 19 other characters from alphabet and one image from each one
        # add data to test_episodes


if __name__ == '__main__':
    path_to_alphabets = sys.argv[1]
    main(path_to_alphabets)
