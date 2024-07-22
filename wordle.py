#!/usr/bin/env python3

import numpy as np


def make_pattern(solution, guess):
    """
    Returns the pattern from a solution if a guess is made.

    Parameters:
    ===========
    solution: [..., 5], broadcastable to guess.
    guess: [..., 5], broadcastable to solution.

    Returns:
    ========
    pattern: broadcastable shape from solution.shape[:-1] and guess.shape[:-1].
             The patterns are encoded as uint8 base 3, with the highest
             significant digit representing the most left pattern colour and
             the lowest significant digit representing the most right pattern
             colour. The digits are
             - 0 for black, guess letter is not in solution
             - 1 for yellow, guess letter is in solution, but at a different
               location
             - 2 for green, guess letter is at the correct location in
               solution.
    """
    exps = np.uint8(3) ** np.arange(4, -1, -1, dtype=np.uint8)

    d = guess[..., :, None] == solution[..., None, :]

    # Match first along the diagonal: matching letters at the correct location
    matches = np.diagonal(d, axis1=-2, axis2=-1)
    pattern = np.sum(
        exps * matches.astype(np.uint8) * np.uint8(2),
        axis=-1,
        dtype=np.uint8
    )
    # Disable matching solution and guess letters that have been matched
    d &= ~matches[..., None, :] & ~matches[..., :, None]

    # Process guess letters that are in the solution letters
    m = np.ones(shape=d.shape[:-2] + d.shape[-1:], dtype=bool)
    for i in range(5):
        j = np.argmax(d[..., i, :], axis=-1, keepdims=True)
        matches = np.take_along_axis(d[..., i, :], j, axis=-1)

        pattern += exps[i] * np.squeeze(matches, axis=-1).astype(np.uint8)

        m[...] = True
        np.put_along_axis(m, j, ~matches, axis=-1)
        d &= m[..., None, :]

    return pattern


def calcuate_entropy(patterns):
    """
    Parameters:
    ===========
    patterns: [S, G]

    Returns:
    ========
    entropy: [G]
    """
    indexing = (
        np.ones([len(patterns), 1], dtype=np.uint16)
        * np.arange(len(corpus), dtype=np.uint16)
    )  # [S, G]
    counts = np.zeros([3 ** 5, len(corpus)], np.uint16)
    np.add.at(counts, (patterns, indexing), 1)

    probs = counts.astype(np.float16) / len(patterns)
    log_probs = np.log(probs, out=np.zeros_like(probs), where=counts > 0)
    return -np.sum(probs * log_probs, axis=0)


if __name__ == '__main__':
    with open('words', 'r') as f:
        corpus = [w[:-1] for w in f.readlines()]
        corpus = [w for w in corpus if len(w) == 5 and w.isalpha()]
        corpus = np.array(corpus, dtype=str)
        corpus = corpus.view('U1').reshape((-1, 5))

    solution_indexes = np.arange(len(corpus))
    patterns = None
    while len(solution_indexes) > 1:
        if patterns is None:
            guess = np.array(['tares']).view('U1')
            guess_index = np.argmax(np.all(corpus == guess, axis=-1))
        else:
            entropy = calcuate_entropy(patterns)
            guess_indexes, = np.where(entropy == np.max(entropy))
            guess_in_solutions = np.in1d(guess_indexes, solution_indexes)
            guess_index = guess_indexes[np.argmax(guess_in_solutions)]

        guess = np.squeeze(corpus[guess_index].view('U5'))
        while True:
            pattern = input(f'guess={guess} - pattern=')
            if len(pattern) in (0, 5):
                break

        if len(pattern) == 0:
            corpus = np.delete(corpus, guess_index, axis=0)

            patterns = patterns[solution_indexes != guess_index]
            patterns = np.delete(patterns, guess_index, axis=1)

            solution_indexes = solution_indexes[solution_indexes != guess_index]
            solution_indexes[solution_indexes > guess_index] -= 1

        else:
            pattern = int(pattern, base=3)
            patterns_given_guess = (
                make_pattern(corpus, corpus[guess_index])
                if patterns is None
                else patterns[:, guess_index]
            )
            mask = patterns_given_guess == pattern
            solution_indexes = solution_indexes[mask]

            patterns = (
                make_pattern(corpus[solution_indexes, None], corpus[None, :])
                if patterns is None
                else patterns[mask]
            )

        print(f'solutions={len(solution_indexes)}')

    solution = np.squeeze(corpus[solution_indexes].view('U5'))
    print(f'solution={solution}')
