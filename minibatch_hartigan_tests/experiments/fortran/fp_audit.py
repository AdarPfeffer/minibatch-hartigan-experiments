"""fp_audit.py — prove fp_as136_dp.f and fp_as136_sp.f are the same program at two widths.

The float64-vs-float32 experiment only means something if the two variants differ ONLY in
floating-point width. A `diff` cannot show that on its own, because the intended differences
(type keywords, literal suffixes, the _dp/_sp renames needed to link both into one library)
look the same as an accidental one would.

So this inverts the generator's transformation on the float32 file and requires the result to
be byte-identical to the float64 file. Any real difference — a changed comparison, a dropped
line, a reordered loop — survives the inversion and fails the check.

Exit 0 = the two variants are provably the same program. Run via `make fp-audit`.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def invert(sp):
    """Undo to_single() from gen_fp_variants.py, then the _sp -> _dp renames."""
    t = sp.replace("real          ", "double precision")
    t = t.replace("0.0E+00", "0.0D+00").replace("1.0E+00", "1.0D+00")
    t = t.replace("1.0E+30", "1.0D+30")
    t = re.sub(r"\breal \(", "dble (", t)
    t = re.sub(r"\br4_huge_sp\b", "r8_huge_dp", t)
    t = re.sub(r"\breal r8_huge_dp\b", "double precision r8_huge_dp", t)
    t = t.replace("float32 (real)", "float64 (double precision)")
    t = t.replace("fp_as136_sp.f", "fp_as136_dp.f")
    return re.sub(r"_sp\b", "_dp", t)


def main():
    dp_path = os.path.join(HERE, "fp_as136_dp.f")
    sp_path = os.path.join(HERE, "fp_as136_sp.f")
    for p in (dp_path, sp_path):
        if not os.path.exists(p):
            sys.exit(f"FP-AUDIT FAILED — {os.path.basename(p)} missing; run gen_fp_variants.py")

    dp = open(dp_path).read().splitlines()
    sp_raw = open(sp_path).read().splitlines()      # as it will be compiled
    sp = invert("\n".join(sp_raw)).splitlines()     # for the equivalence check

    if len(dp) != len(sp):
        sys.exit(f"FP-AUDIT FAILED — line counts differ: dp={len(dp)} sp={len(sp)}")

    bad = [(i + 1, a, b) for i, (a, b) in enumerate(zip(dp, sp)) if a != b]
    if bad:
        print(f"FP-AUDIT FAILED — {len(bad)} line(s) differ beyond precision:")
        for i, a, b in bad[:20]:
            print(f"  line {i}\n    dp: {a}\n    sp: {b}")
        sys.exit(1)

    # FIXED-FORM COLUMN LIMIT. Fortran ignores everything past column 72 and does so SILENTLY,
    # so an over-long line is not a compile error where it occurs — it corrupts the statement
    # and surfaces somewhere else entirely. The _dp/_sp rename adds 3 characters to every
    # renamed symbol, which is exactly how `d,` fell off the KMNS signature and fused with the
    # next line's `itran` into `ditran`. Comments are exempt (as136.f ships 20 long ones).
    for path, lines in (("fp_as136_dp.f", dp), ("fp_as136_sp.f", sp_raw)):
        over = [(i + 1, len(l), l) for i, l in enumerate(lines)
                if l[:1].lower() not in ("c", "*", "!") and len(l) > 72]
        if over:
            print(f"FP-AUDIT FAILED — {path}: {len(over)} code line(s) past column 72; "
                  f"fixed-form Fortran will truncate them silently:")
            for i, n, l in over[:10]:
                print(f"  line {i} ({n} cols): {l}")
            sys.exit(1)

    # code lines only — fixed-form Fortran puts 'c' in column 1 for comments, and the file
    # header mentions fptrc_add in prose.
    n_trace = sum(1 for line in dp
                  if not line[:1].lower() == "c" and "call fptrc_add" in line)
    if n_trace != 2:
        sys.exit(f"FP-AUDIT FAILED — expected 2 trace call sites (OPTRA, QTRAN), found {n_trace}")

    print(f"FP-AUDIT OK — {len(dp)} lines, identical after inverting the precision transform; "
          f"2 trace call sites; the two variants are the same program at two widths.")


if __name__ == "__main__":
    main()
