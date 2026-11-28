# Changes

All notable changes of PyAstroDyn-edu are listed here, the most recent first.
Function names may still change before version 0.1.0.

## 0.0.13 (November 2026)

- README, section 9: the use of AI is now disclosed with the line "AI use:
  code-by-intent" and the DOI of the note that describes it
  (https://doi.org/10.5281/zenodo.24626373).
- No change of the functions and of the example problems.

## 0.0.12 (November 2026)

- README corrected: the table "What is covered" and the first example no
  longer present the two-impulse coplanar transfers (Hohmann, bi-elliptic) as
  part of Lecture 6. They are taught in Lecture 7.
- No change of the functions and of the example problems.

## 0.0.11 (November 2026)

- Lecture 6 added, module `maneuvers`: rocket equation, impulsive burns,
  single-impulse maneuvers, tangential impulses at the apses, burn of finite
  duration, and the functions of the two-impulse coplanar transfers
  (Hohmann, bi-elliptic), which are taught at the start of Lecture 7: their
  example problems come with Lecture 7.
- Ten example problems of Lecture 6 (`examples/lecture06`), each as a
  script and as a notebook, and the notebook
  `notebooks/Lecture06_maneuvers_1.ipynb`.
- No change of the functions of Lectures 1 to 5.

## 0.0.10 (November 2026)

- DOI of the library (all versions) added to the README and the citation file.
- README: where to find the library (GitHub, PyPI, Zenodo) and how to cite it.
- `.zenodo.json` added, so that Zenodo takes the author with ORCID, the license
  and the keywords from it.
- "Licence" spelled "license" throughout, as in the MIT License.
- README: a picture of typical Earth orbits computed with the library
  (`docs/thumbnail.png`, made by `tools/make_thumbnail.py`).
- No change of the functions.

## 0.0.9 (November 2026)

First version distributed outside the author's group.

- Part 1 of the course, Lectures 1 to 5: the two-body equation of motion and
  its integrals, numerical integration, conic sections, elliptical orbits,
  anomalies and Kepler's equation, rotation matrices, orbital elements and
  Lagrange coefficients; units and scaling.
- Example problems for every lecture, each as a script and as a notebook,
  with stored results and an independent check.
- Documents generated from the code: function guide, problems, cross-reference
  with other tools; a getting-started guide.
