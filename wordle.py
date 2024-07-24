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
    log_probs = -np.log(probs, out=np.zeros_like(probs), where=counts > 0)
    return np.sum(probs * log_probs, axis=0)


def filter_solutions(solutions, guess, pattern):
    """
    Computes a mask over solutions that are possible given that a guess
    resulted in a pattern.

    Parameters:
    ===========
    solutions: [S, 5]
    guess: [5]
    pattern: []

    Returns:
    ========
    mask: [S]
    """
    patterns_given_guess = make_pattern(solutions, guess)
    return patterns_given_guess == pattern


if __name__ == '__main__':
    print("Wordle solver!")

    with open('official_wordle_all.txt', 'r') as f:
        corpus = [w[:-1] for w in f.readlines()]
        corpus = [w.lower() for w in corpus if len(w) == 5 and w.isalpha()]

    with open('official_wordle_common.txt', 'r') as f:
        solutions = [w[:-1] for w in f.readlines()]
        solutions = {w.lower()
                     for w in solutions if len(w) == 5 and w.isalpha()}

    solution_mask = np.array([
        word in solutions for word in corpus
    ])

    corpus = np.array(corpus, dtype=str)
    corpus = corpus.view('U1').reshape((-1, 5))

    patterns = None
    while True:
        if patterns is None:
            suggested_guess = np.array(['soare']).view('U1')

        else:
            entropy = calcuate_entropy(patterns)
            sorted_indexes = np.lexsort((solution_mask, entropy))[::-1]
            suggested_guess = corpus[sorted_indexes[0]]

            num_solutions = np.count_nonzero(solution_mask)
            sorted_solution_indexes = sorted_indexes[
                solution_mask[sorted_indexes]
            ]
            solutions_fmt = [
                f"{str(np.squeeze(corpus[index].view('U5'))).upper()}"
                for index in sorted_solution_indexes[:5]
            ]
            entropy_fmt = [
                f"{entropy[index]:.2f}"
                for index in sorted_solution_indexes[:5]
            ]
            solution_list_fmt = ', '.join([
                f"{s} ({e})"
                for s, e
                in zip(solutions_fmt, entropy_fmt)
            ])
            cont = '…' if len(sorted_solution_indexes) > 10 else ''
            print(f"Possible solutions (n={num_solutions}): "
                  f"{solution_list_fmt} {cont}")

            guesses_fmt = [
                f"{str(np.squeeze(corpus[index].view('U5'))).upper()}"
                for index in sorted_indexes[:5]
            ]
            entropy_fmt = [
                f"{entropy[index]:.2f}"
                for index in sorted_indexes[:5]
            ]
            guess_list_fmt = ', '.join([
                f"{s} ({e})"
                for s, e
                in zip(guesses_fmt, entropy_fmt)
            ])
            cont = '…' if len(sorted_indexes) > 10 else ''
            print(f"Top guesses: "
                  f"{guess_list_fmt} {cont}")

        while True:
            suggested_guess_str = str(
                np.squeeze(suggested_guess.view('U5'))
            ).upper()
            guess = input(f"Guess [{suggested_guess_str}] ")
            if len(guess) == 0:
                guess = suggested_guess
                break
            if len(guess) == 5 and guess.isalpha():
                guess = np.array([guess.lower()]).view('U1')
                break
            print("Enter a guess you would like to make, leaving it blank "
                  "to accept the suggested guess in the brackets.")
        while True:
            pattern = input("Pattern ")
            if len(pattern) == 5 and all(ch in '012' for ch in pattern):
                pattern = int(pattern, base=3)
                break
            print("Enter a pattern as a 5-digit code with 0=black, 1=yellow "
                  "2=green, e.g. 01200=black-yellow-green-black-black")

        if pattern == int('22222', base=3):
            break

        mask = filter_solutions(corpus[solution_mask], guess, pattern)
        solution_mask[solution_mask] = mask
        patterns = (
            make_pattern(corpus[solution_mask, None], corpus[None, :])
            if patterns is None
            else patterns[mask]
        )

        print()
