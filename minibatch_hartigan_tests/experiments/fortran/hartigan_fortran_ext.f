c hartigan_fortran_ext.f — small extension to as136.f for the best-of-N init benchmark.
c
c Adds, WITHOUT changing the kmns/optra/qtran signatures (so the legacy C driver is
c unaffected): a live-set toggle (g_uselive) and a warning-verbosity flag (g_verbose),
c both in common /hgn_cfg/, plus two run outputs (g_niter, g_postinit, common /hgn_out/)
c set inside kmns. All accessed from Python ctypes via the tiny subroutines below.
c BLOCK DATA defaults g_uselive=1 (so an un-set run reproduces AS-136 exactly) and
c g_verbose=0 (silent: the max_iters=1 epoch-2 timing replays would otherwise flood
c stdout with non-convergence warnings; IFAULT=2 is still returned to the caller).
c
      block data hgn_init
      integer g_uselive
      integer g_verbose
      common /hgn_cfg/ g_uselive, g_verbose
      data g_uselive /1/
      data g_verbose /0/
      end

      subroutine hartigan_set_uselive ( v )
      integer v
      integer g_uselive
      integer g_verbose
      common /hgn_cfg/ g_uselive, g_verbose
      g_uselive = v
      return
      end

      subroutine hartigan_set_verbose ( v )
      integer v
      integer g_uselive
      integer g_verbose
      common /hgn_cfg/ g_uselive, g_verbose
      g_verbose = v
      return
      end

      subroutine hartigan_get_niter ( out )
      integer out
      integer g_niter
      double precision g_postinit
      common /hgn_out/ g_niter, g_postinit
      out = g_niter
      return
      end

      subroutine hartigan_get_postinit ( out )
      double precision out
      integer g_niter
      double precision g_postinit
      common /hgn_out/ g_niter, g_postinit
      out = g_postinit
      return
      end
