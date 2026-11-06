# AS 136 Hartigan-Wong (Fortran 77) shared libraries, loaded from Python through ctypes.
#   make lib       -> libas136.so          (hartigan_fortran_ctypes.py; impls fortran_ls / fortran_nols)
#   make lib-fp    -> libfp136.so          (fp_hartigan_ctypes.py; float64-vs-float32 study)
#   make fp-audit  -> equivalence audit of the generated float64/float32 sources
#   make lib-ref   -> libas136_ref{64,32}.so (fp_untouched_check.py; untouched-source control)
# Flags are exactly those the shipped data were produced with.

FC      = gfortran
FFLAGS  = -O3 -std=legacy -ffixed-form -w

# Shared library for the best-of-N init benchmark (Python ctypes: hartigan_fortran_ctypes.py).
# Bundles as136.f (kmns/optra/qtran) + hartigan_fortran_ext.f (toggle + niter/postinit getters).
lib: libas136.so
libas136.so: as136.f hartigan_fortran_ext.f
	$(FC) $(FFLAGS) -fPIC -shared as136.f hartigan_fortran_ext.f -o libas136.so

# float64-vs-float32 AS-136 comparison (fp_precision_compare.py). Both variants are
# GENERATED from as136.f by gen_fp_variants.py so they are provably line-for-line identical
# apart from the floating-point width; regenerate before building if as136.f ever changes.
# Production libas136.so is untouched by this target.
lib-fp: libfp136.so
fp_as136_dp.f fp_as136_sp.f: as136.f gen_fp_variants.py
	python3 gen_fp_variants.py
libfp136.so: fp_as136_dp.f fp_as136_sp.f fp_trace.f hartigan_fortran_ext.f
	$(FC) $(FFLAGS) -fPIC -shared fp_as136_dp.f fp_as136_sp.f fp_trace.f \
	    hartigan_fortran_ext.f -o libfp136.so

# The audit that makes the experiment meaningful: inverts the precision transform on the
# float32 file and requires an exact match with the float64 one. A plain diff cannot do this
# — the intended differences look like accidental ones.
fp-audit: fp_as136_dp.f fp_as136_sp.f
	python3 fp_audit.py

# UNTOUCHED-SOURCE CONTROL (fp_untouched_check.py). Compiles as136.f twice, unmodified:
# once normally (float64) and once with -freal-8-real-4, which demotes every `double
# precision` in the source to 4-byte real AT COMPILE TIME. No generator, no rename, no trace
# call, no re-wrap — not one character of as136.f changes. This is the control that removes
# gen_fp_variants.py from the chain entirely.
lib-ref: libas136_ref64.so libas136_ref32.so
libas136_ref64.so: as136.f hartigan_fortran_ext.f
	$(FC) $(FFLAGS) -fPIC -shared as136.f hartigan_fortran_ext.f -o $@
libas136_ref32.so: as136.f hartigan_fortran_ext.f
	$(FC) $(FFLAGS) -freal-8-real-4 -fPIC -shared as136.f hartigan_fortran_ext.f -o $@

.PHONY: lib lib-fp lib-ref fp-audit clean

clean:
	rm -f *.o libas136.so
	rm -f libfp136.so fp_as136_dp.f fp_as136_sp.f libas136_ref64.so libas136_ref32.so
