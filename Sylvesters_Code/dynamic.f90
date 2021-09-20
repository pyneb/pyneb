!------program for trajectory calculation in SF------
!------based on principle of least action path-------
 module input_in     
  implicit real*8 (a-h,o-z)
  double precision,dimension(:),  allocatable :: q20,q30
  double precision,dimension(:,:),allocatable :: v,e0
  integer q20max, q30max
end module input_in

program main
  use input_in
  implicit real*8 (a-h,o-z)
  integer, dimension(:,:), allocatable :: ip
  double precision, dimension(:  ), allocatable :: q0nd,q2nd,send
  double precision, dimension(:,:), allocatable :: s
  double precision, dimension(:  ), allocatable :: a

  alrg=1000.0
  alrg2=500.0
  notl=1000

  open(10,file='input.dat',status='old')
  open(20,file='POT_2.out',status='old')
  open(30,file='result.out',status='unknown')

  read(10,*)q20max,q30max,zero_en,lspan,jinitial

  allocate(q20(q20max))
  allocate(q30(q30max))
  allocate( v(q20max,q30max))
  allocate(e0(q20max,q30max))

  allocate(ip(q20max,q30max))
  allocate( s(q20max,q30max))
  allocate(   a(q30max))
  allocate(q0nd(notl))
  allocate(q2nd(notl))
  allocate(send(notl))  

  do i=1,q20max
     do j=1,q30max
        read(20,*)q20(i),q30(j),v(i,j)
     enddo
  enddo
  write(*,*)'reading completed'

  dq20=q20(2)-q20(1)
  dq30=q30(2)-q30(1)

!-----calculation of vmin, xmin, and ymin and starting point ---------
  vmin=1000.0
  do i=1,q20max
     do j=1,q30max
        if(v(i,j).lt.vmin)then
           vmin=v(i,j)
           xmin=q20(i)
           ymin=q30(j)
           istart=i
           jstart=j
        endif
     enddo
  enddo
  write(*,*)"vmin, xmin, ymin, istart, jstart",vmin,xmin,ymin,istart,jstart

!----------scaling the energy -----------------
  do i=1,q20max
     do j=1,q30max
        v(i,j)=v(i,j) - vmin
     enddo
  enddo   

!-----calculation of end point of the trajectories--------
  vmin_b=1000.0
  do i=((q20max+1)/2),q20max
     do j=1,q30max
        if(v(i,j).lt.vmin_b)then
           vmin_b=v(i,j)
           xmin_b=q20(i)
           ymin_b=q30(j)
           iend=i
           jend=j
        endif
     enddo
  enddo
  write(*,*)"vmin_b, xmin_b, ymin_b, iend, jend",vmin_b,xmin_b,ymin_b,iend,jend

!-----calculation of least action trajectories ------------------
!-----1. calculation of s for istart---------------------------
 e=0.0d0
 k=istart+1
 do j=1,q30max
    if(v(k,j).gt.e.and.abs(j-jstart).le.2*lspan)then
       call action(e,jstart,j,k,actn)
       s(k,j)=actn
       ip(k,j)=jstart
    else
       s(k,j)=alrg
       ip(k,j)=jstart
    endif
    write(*,99)q20(k),q30(j),s(k,j)
 enddo
 !write(*,*)'e',e
 !stop
!-----2. calculation of s upto iend--------------
  write(30,*)'x-exit **** y-exit **** min-action ****'
 smn0=alrg
 do k=istart+2,iend
    do j=1,q30max
       if(v(k,j).ge.e)then
          if(j.gt.lspan)then
             ll1=lspan
          else
             ll1=j-1
          endif
          if(q30max-j.gt.lspan)then
             ll2=lspan
          else
             ll2=q30max-j
          endif
          lk=0
          do l=j-ll1,j+ll2
             if(v(k-1,l).ge.e.and.s(k-1,l).lt.alrg2)then
                lk=lk+1
                call action(e,l,j,k,actn)
                a(l)=actn+s(k-1,l)
             else
                a(l)=alrg
             endif
          enddo
          s(k,j)=a(j+ll2)
          ip(k,j)=j+ll2
          do l=j-ll1,j+ll2-1
             if(a(l).lt.s(k,j))then
                s(k,j)=a(l)
                ip(k,j)=l
             endif
          enddo
       else
          s(k,j)=alrg
          ip(k,j)=ip(k-1,j)
       endif
       write(*,99)q20(k),q30(j),s(k,j)
!------ extract nim. action -----------------
       mn=1
       if(k.ge.iend)then
          do i=1,mn
             if(iend.eq.k.and.jend.eq.j)then
                q0nd(i)=q20(k)
                q2nd(i)=q30(j)
                send(i)=s(k,j)
                if(s(k,j).lt.smn0)then
                   smn0=s(k,j)
                   k0=k
                   j0=j
                endif
             endif
          enddo
       endif
!--------------------------------------------
    enddo
 enddo
 write(30,*)'****trajectory for min-action =',s(k0,j0)
 write(30,*)'***  Q20  ***   Q30     ***     Veff    ***   path length(scaling required)'
 m=j0
 scount=0.0d0
 do ll=k0,istart,-1
    q20r=q20(ll)
    q30r=q30(m)
    veff=v(ll,m)
    if(ll.lt.k0)then
       scount=scount+dsqrt((q20f-q20r)**2+(q30f-q30r)**2)
    endif
    q20f=q20r
    q30f=q30r
    write(30,95)q20r,q30r,veff,-scount
    m=ip(ll,m)
 enddo

 99   format(1x,2f10.3,f12.4)
 95   format(1x,2f10.3,2f18.5)

     stop
end program  


subroutine action(e,ji,jf,if,actn)
 use input_in
 implicit real*8 (a-h,o-z)
 dimension vv(1000) 
 q20i=q20(if-1)
 q20f=q20(if)
 q30i=q30(ji)
 q30f=q30(jf)
 r=dsqrt((q20f-q20i)**2+(q30f-q30i)**2)
 theta=atan((q30f-q30i)/(q20f-q20i))
 if(r.le.1.0)then 
  nr=10
 elseif(r.gt.1.0.and.r.le.2.0)then
  nr=10
 elseif(r.gt.2.0.and.r.le.3.0)then
  nr=20
 elseif(r.gt.3.0.and.r.le.4.0)then
  nr=30
 elseif(r.gt.4.0)then
  nr=50
 endif 
 dr=r/dfloat(nr)
 do i=1,nr+1
  x=q20i+dr*cos(theta)*dfloat(i-1)
  y=q30i+dr*sin(theta)*dfloat(i-1)
 enddo
 actn=0.0d0 
 do i=2,nr,2
  v1=vv(i-1)
  v2=vv(i)
  v3=vv(i+1)
  dx=dr*cos(theta)
  dy=dr*sin(theta)  
  arg=dsqrt(2.0d0*(v2-e))
  if(arg.gt.0.0d0)then 
   s=(dr*(arg))
  else
   s=0.0d0
  endif
  actn=actn+s
 enddo
 return
 end subroutine 
