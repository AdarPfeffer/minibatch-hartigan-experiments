c fp_trace.f — decision recorder shared by fp_as136_dp.f and fp_as136_sp.f.
c
c The float64/float32 experiment asks whether the two variants make the SAME DECISIONS,
c not merely whether they end up in the same place. So each committed transfer appends
c one record:
c
c     (stage, i, l1, l2)   stage 1 = OPTRA, 2 = QTRAN;  point i moves from l1 to l2
c
c This is the only code added to AS-136. It reads no floating-point value, takes no branch
c on one, and consumes no RNG — so both variants execute exactly the arithmetic AS-136
c specifies, and the trace is a pure observation of what they decided.
c
c One buffer serves both variants: the driver calls fptrc_reset before each run and
c fptrc_get after it, so the float64 and float32 traces never coexist. If a run exceeds
c FPTMAX transfers the overflow flag is set and recording stops — the comparison is then
c reported as TRUNCATED rather than silently short.
c
      block data fptrc_data
      integer FPTMAX
      parameter ( FPTMAX = 2000000 )
      integer g_ntrc
      integer g_ovf
      integer g_trc(4,FPTMAX)
      common /fptrc/ g_ntrc, g_ovf, g_trc
      data g_ntrc /0/
      data g_ovf /0/
      end

      subroutine fptrc_reset ( )
      integer FPTMAX
      parameter ( FPTMAX = 2000000 )
      integer g_ntrc
      integer g_ovf
      integer g_trc(4,FPTMAX)
      common /fptrc/ g_ntrc, g_ovf, g_trc
      g_ntrc = 0
      g_ovf = 0
      return
      end

      subroutine fptrc_add ( stage, i, l1, l2 )
      integer stage
      integer i
      integer l1
      integer l2
      integer FPTMAX
      parameter ( FPTMAX = 2000000 )
      integer g_ntrc
      integer g_ovf
      integer g_trc(4,FPTMAX)
      common /fptrc/ g_ntrc, g_ovf, g_trc
      if ( g_ntrc .ge. FPTMAX ) then
        g_ovf = 1
        return
      end if
      g_ntrc = g_ntrc + 1
      g_trc(1,g_ntrc) = stage
      g_trc(2,g_ntrc) = i
      g_trc(3,g_ntrc) = l1
      g_trc(4,g_ntrc) = l2
      return
      end

      subroutine fptrc_count ( out )
      integer out
      integer FPTMAX
      parameter ( FPTMAX = 2000000 )
      integer g_ntrc
      integer g_ovf
      integer g_trc(4,FPTMAX)
      common /fptrc/ g_ntrc, g_ovf, g_trc
      out = g_ntrc
      return
      end

      subroutine fptrc_overflow ( out )
      integer out
      integer FPTMAX
      parameter ( FPTMAX = 2000000 )
      integer g_ntrc
      integer g_ovf
      integer g_trc(4,FPTMAX)
      common /fptrc/ g_ntrc, g_ovf, g_trc
      out = g_ovf
      return
      end

c  Copy the first min(nmax, g_ntrc) records into buf(4, nmax); returns how many.
      subroutine fptrc_get ( buf, nmax, ngot )
      integer nmax
      integer ngot
      integer buf(4,nmax)
      integer FPTMAX
      parameter ( FPTMAX = 2000000 )
      integer g_ntrc
      integer g_ovf
      integer g_trc(4,FPTMAX)
      common /fptrc/ g_ntrc, g_ovf, g_trc
      integer t
      integer q
      ngot = min ( nmax, g_ntrc )
      do t = 1, ngot
        do q = 1, 4
          buf(q,t) = g_trc(q,t)
        end do
      end do
      return
      end
