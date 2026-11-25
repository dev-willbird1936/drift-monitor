# Stage 2 paper

EMNLP / ACL template (`acl.sty`, `acl_natbib.bst` from github.com/acl-org/acl-style-files), `final` mode (named, not anonymous).
Main body is under 6 pages; the appendix follows the references.

Build:

```bash
python make_figures.py                         # fig_loop.pdf (numbers copied from ../README.md)
tectonic -X compile main.tex                     # main.pdf
```

Overleaf also works: upload `main.tex`, `refs.bib`, `acl.sty`, `acl_natbib.bst`, `fig_loop.pdf`.

Stage 2 form (https://tally.so/r/4aKyKA, due 28 Nov 23:59 AoE) takes `main.pdf` as the preprint upload, plus the two Hugging Face collection links and the GitHub link. The form's checklist asks that the toolkits and models used are cited: AlignTune, CuratorKIT, AuditKIT and Tiny Aya are in `refs.bib`.

Confirm before submitting: the Ethics Statement says the recorded sessions hold no human users' conversations or third-party data.
