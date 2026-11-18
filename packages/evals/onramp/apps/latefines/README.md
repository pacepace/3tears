# latefines

Overdue fines for the Eastgate branch library. `fines_file(returns, closures)` reads a returns file (one loan per
line: `loan_id`, `item_type`, `reader`, `due`, `returned`) and the branch's closures file, and gives every loan's fine
in pence; `fine(loan, closures)` does one loan. The rules are in the module docstring of `fines.py`:

- every day after the due date, up to and including the day the item comes back, is a late day;
- a late day costs the item type's daily rate (books and audiobooks 25p, magazines 10p, DVDs £1, laptops £5) unless
  the library was closed that day: every Sunday, and every day of each closure in `data/closures.jsonl`, from its
  `first` day to its `last` day, both included (a bank holiday is a closure whose first and last day are the same);
- a loan's fine is capped per item type (books and audiobooks £8, magazines £2, DVDs £12, laptops £50);
- only adult readers are fined; junior and concession readers pay nothing.

`data/returns.jsonl` is the spring-term and winter-break returns we checked by hand when the old library system was
switched off, in the order they came back. `fine_pence` on each line is what the old system charged, which the desk
checked against the rules above. The new script should charge every loan exactly that.

Tests: `python -m unittest discover tests` from this directory.
