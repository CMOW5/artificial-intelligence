import os
import random
import math
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    print('sum = ', sum(ranks.values()))

    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")

    print('sum = ', sum(ranks.values()))


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, from_page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability_distribution = {}

    for to_page in corpus:
        probability_distribution[to_page] = probability_factor(corpus, from_page, to_page, damping_factor) + \
                                            additional_probability_factor(corpus, from_page, damping_factor)
    return probability_distribution


def probability_factor(corpus, from_page, to_page, damping_factor):
    outgoing_links = corpus[from_page]
    total_outgoing_pages = len(outgoing_links)

    # if current page has not outgoing links, then return a probability distribution that chooses randomly among
    # all pages with equal probability
    if total_outgoing_pages == 0:
        return 1 / len(corpus)

    # no outgoing link from current_page to to_page
    if to_page not in outgoing_links:
        return 0

    return damping_factor / total_outgoing_pages


def additional_probability_factor(corpus, from_page, damping_factor):
    outgoing_links = corpus[from_page]
    total_outgoing_pages = len(outgoing_links)

    # if current page has not outgoing links, then no need to add the extra additional factor
    if total_outgoing_pages == 0:
        return 0

    return (1 - damping_factor) / len(corpus)


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    page_ranks = {page: 0 for page in corpus}

    # The first sample should be generated by choosing from a page at random
    chosen_page = random.choice(list(corpus.keys()))
    page_ranks[chosen_page] += 1 / n

    for i in range(n - 1):
        # For each of the remaining samples, the next sample should be generated from the previous sample based on the
        # previous sample’s transition model
        probability_distribution = transition_model(corpus, chosen_page, damping_factor)

        all_pages = list(probability_distribution.keys())
        weights = list(probability_distribution.values())
        chosen_page = random.choices(all_pages, weights)[0]

        page_ranks[chosen_page] += 1 / n

    return page_ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    total_number_of_pages = len(corpus)
    page_ranks = {page: {"rank": 1 / total_number_of_pages, "delta": math.inf} for page in corpus}
    accuracy = 0.001

    while should_continue(page_ranks, accuracy):
        for page in page_ranks.keys():
            previous_page_rank = page_ranks[page]["rank"]
            current_page_rank = page_rank(corpus, damping_factor, page, page_ranks)
            delta = abs(current_page_rank - previous_page_rank)

            page_ranks[page]["rank"] = current_page_rank
            page_ranks[page]["delta"] = delta

    return {page: page_ranks[page]["rank"] for page in page_ranks}


def should_continue(page_ranks, accuracy):
    for page in page_ranks:
        if page_ranks[page]["delta"] > accuracy:
            return True
    return False


def page_rank(corpus, damping_factor, current_page, page_ranks):
    total_number_of_pages = len(corpus)
    pages_that_links_to_current_page = get_pages_that_link_to_page(corpus, current_page)

    page_rank_sum = 0
    for linking_page in pages_that_links_to_current_page:
        # A page that has no links at all should be interpreted as having one link
        # for every page in the corpus (including itself).
        number_of_links_on_linking_page = len(corpus[linking_page]) if len(corpus[linking_page]) != 0 else len(corpus)

        page_rank_sum += page_ranks[linking_page]["rank"] / number_of_links_on_linking_page

    return ((1 - damping_factor) / total_number_of_pages) + (damping_factor * page_rank_sum)


def get_pages_that_link_to_page(corpus, page):
    pages_that_links_to_page = set()

    for containing_page in corpus:
        if page in corpus[containing_page]:
            pages_that_links_to_page.add(containing_page)

    return pages_that_links_to_page


if __name__ == "__main__":
    main()
