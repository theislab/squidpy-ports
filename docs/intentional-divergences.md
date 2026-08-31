# Intentional divergences from STalign

One, and it is algorithmic rather than per-notebook. At rank 3 upstream's regularisation *energy*
transforms two of three spatial axes ([`STalign.py:1504`](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/STalign/STalign.py#L1504)) -- byte-identical to the
rank-2 line at [`:1193`](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/STalign/STalign.py#L1193), where two axes is all of them -- while the gradient it
descends smooths all three ([`:1527`](https://github.com/JEFworks-Lab/STalign/blob/b2068edc98974efa54537eca194736e177bbe11d/STalign/STalign.py#L1527)): a rank-2 line reused without extending it to
the new axis. squidpy transforms all three, which moves the fitted velocity field by 31x and is
why `sigmaR` is retuned from upstream's `1e8` to `1e6`. Unverified against the paper
([Clifton et al. 2023](https://doi.org/10.1038/s41467-023-43915-7)), and pinned by a strict xfail
in `tests/test_reference.py` rather than asserted away.
