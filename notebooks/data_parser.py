"""Legacy entry point; new code lives in russian_rates.

Install first: pip install -e .
Then use: python -m russian_rates --series all
"""

from russian_rates.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
